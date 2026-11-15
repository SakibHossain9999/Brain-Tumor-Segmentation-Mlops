"""The upload service: segment uploaded brain scans and build a 3D viewer bundle from the result.

    PYTHONPATH=pipeline uvicorn api.main:app --port 8000
    then open http://localhost:8000/docs for an interactive page listing (and trying) every endpoint

Endpoints
    GET  /api/health              is the service up, which models are available
    GET  /api/models              the models, their test scores and CPU times, which one is the champion
    POST /api/segment             upload scans (+ optional expert outline) -> a job id
    POST /api/jobs/{id}/models    run one more model on an earlier upload, add it to its result
    GET  /api/jobs/{id}           progress of a job; when done, where its result is
    GET  /api/cases/{id}/{file}   the result: a bundle the viewer opens like a gallery case

A segmentation takes 10-60 s, too long for one web request, so uploads become jobs: the request returns
at once with a job id, a background worker does the work one job at a time, and the page polls for
progress. Uploaded scans are kept with the result, so more models can be run on them later, and both are
deleted together after a few hours.
"""
import gc
import json
import os
import queue
import secrets
import shutil
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "pipeline")]
from api.uploads import UploadError, load_upload  # noqa: E402
from brainviz.build import add_segmentation, build_bundle  # noqa: E402
from brainviz.io import MSD_CHANNELS  # noqa: E402
from ml.fingerprint import identify  # noqa: E402
from ml.inference import load_onnx, predict  # noqa: E402
from ml.metrics import region_metrics  # noqa: E402

# --- settings (environment variables, so the Docker image can change them without code changes)
MODELS_DIR = Path(os.environ.get("BRAIN_MODELS_DIR", ROOT / "models"))
JOBS_DIR = Path(os.environ.get("BRAIN_JOBS_DIR", ROOT / "output" / "jobs"))
WEB_DIR = Path(os.environ.get("BRAIN_WEB_DIR", ROOT / "web" / "dist"))  # the built viewer, served at / if present
FINGERPRINTS = Path(os.environ.get("BRAIN_FINGERPRINTS", ROOT / "splits" / "msd_fingerprints.json"))
THREADS = int(os.environ.get("BRAIN_THREADS", "0")) or None  # CPU cores for the models (None = all)
MAX_UPLOAD_MB = float(os.environ.get("BRAIN_MAX_UPLOAD_MB", "200"))
JOB_TTL_HOURS = float(os.environ.get("BRAIN_JOB_TTL_HOURS", "6"))
ALLOWED_ORIGINS = os.environ.get("BRAIN_ALLOWED_ORIGINS", "*").split(",")  # sites allowed to call the API
# models kept in memory between jobs; others (Swin UNETR: ~7 GB once loaded) are loaded for a job and freed after
KEEP_LOADED = os.environ.get("BRAIN_KEEP_LOADED", "segresnet,segformer3d").split(",")

MANIFEST = json.loads((MODELS_DIR / "manifest.json").read_text())
MODEL_NAMES = list(MANIFEST["models"])
REGIONS = ("WT", "TC", "ET")
if THREADS:
    torch.set_num_threads(THREADS)


# --- jobs
@dataclass
class Job:
    id: str
    models: list
    status: str = "queued"  # queued -> running -> done | failed
    step: str = "Waiting for its turn"
    progress: float = 0.0
    created: float = field(default_factory=time.time)
    error: str | None = None
    result: dict | None = None
    parent: str | None = None  # set for "one more model" jobs: the upload they add to


jobs: dict[str, Job] = {}
jobs_lock = threading.Lock()
work: queue.Queue = queue.Queue()
_models: dict = {}  # loaded ONNX models, kept between jobs


def update(job: Job, step: str, progress: float, status: str = "running"):
    with jobs_lock:
        job.step, job.progress, job.status = step, round(progress, 3), status


def get_model(name: str):
    """A loaded model: cached if it is small (KEEP_LOADED), otherwise loaded fresh (freed after use)."""
    if name in _models:
        return _models[name]
    model = load_onnx(MODELS_DIR / MANIFEST["models"][name]["onnx"], threads=THREADS)
    if name in KEEP_LOADED:
        _models[name] = model
    return model


def describe_identity(ident: dict | None) -> str:
    if not ident:
        return "A brain from outside our dataset: the models have never seen it."
    return f"This is {ident['case_id']} from the MSD brain tumour dataset ({ident['split']} set): {ident['meaning']}."


def load_job_case(job_dir: Path):
    """The uploaded scans (and outline) of a job, read and checked."""
    inputs = json.loads((job_dir / "input.json").read_text())
    scans = inputs["scans"] if isinstance(inputs["scans"], str) else {k: Path(v) for k, v in inputs["scans"].items()}
    return load_upload(scans, inputs.get("label"), inputs.get("label_format", "auto"))


def run_model(name: str, case, image) -> dict:
    """One model on one brain -> its segmentation entry (regions, timing, and scores if there is an outline)."""
    model, cfg = get_model(name)
    t0 = time.time()
    regions = (predict(model, image, cfg, sw_batch_size=1) > 0.5).cpu().numpy()
    del model  # a model that isn't kept loaded is freed here (its memory goes back to the system)
    gc.collect()
    seg = {
        "regions": regions, "label": MANIFEST["models"][name]["display_name"],
        "model_version": MANIFEST["models"][name]["version"],
        "champion": name == MANIFEST["champion"],
        "seconds": round(time.time() - t0, 1),
    }
    if case.regions is not None:  # an expert outline was uploaded: score the model against it
        scores = {r: {k: round(v, 4) for k, v in region_metrics(regions[i], case.regions[i], list(case.spacing)).items()}
                  for i, r in enumerate(REGIONS)}
        seg["scores"] = scores
        seg["dice_mean"] = round(float(np.mean([scores[r]["dice"] for r in REGIONS])), 4)
    return seg


def run_job(job: Job):
    if job.parent:
        return run_extra_model(job)
    job_dir = JOBS_DIR / job.id
    started = time.time()

    update(job, "Checking the files", 0.03)
    case, info = load_job_case(job_dir)

    update(job, "Checking whether this brain is from our dataset", 0.08)
    ident = identify(case.images, str(FINGERPRINTS)) if FINGERPRINTS.exists() else None
    case.case_id = ident["case_id"] if ident else "upload"  # shown next to the title when recognised

    image = torch.from_numpy(np.stack([case.images[c] for c in MSD_CHANNELS]))  # FLAIR, T1, T1 + contrast, T2
    segmentations = {}
    if case.regions is not None:
        segmentations["expert"] = {"regions": case.regions, "label": "Expert outline"}
    # progress: share the 10-80% range between the models by their expected time
    est = {m: MANIFEST["models"][m].get("cpu_s_per_case_2_threads") or 10 for m in job.models}
    done_est = 0.0
    for name in job.models:
        label = MANIFEST["models"][name]["display_name"]
        update(job, f"Running {label} (about {round(est[name])} s on 2 CPU cores)", 0.1 + 0.7 * done_est / sum(est.values()))
        segmentations[name] = run_model(name, case, image)
        done_est += est[name]

    update(job, "Building the 3D view", 0.82)
    default = MANIFEST["champion"] if MANIFEST["champion"] in job.models else job.models[0]
    description = describe_identity(ident)
    if case.regions is None:
        description += " No expert outline was uploaded, so there are no scores."
    build_bundle(case, job_dir / "bundle", segmentations, default=default, log=lambda *a: None, extra_meta={
        "title": "Your upload",
        "description": description,
        "split": ident["split"] if ident else "unknown",
        "upload": {"format": info["format"], "label_format": info.get("label_format"), "identified": ident,
                   "models": job.models, "seconds": round(time.time() - started, 1)},
    })
    result = {
        "case_url": f"/api/cases/{job.id}",
        "identified": ident,
        "description": description,
        "seconds": round(time.time() - started, 1),
        "models": {m: {k: v for k, v in segmentations[m].items() if k != "regions"} for m in job.models},
    }
    with jobs_lock:
        job.result = result
    update(job, "Done", 1.0, status="done")


def run_extra_model(job: Job):
    """Run one more model on an earlier upload and add it to that upload's result."""
    parent = jobs[job.parent]
    name = job.models[0]
    label = MANIFEST["models"][name]["display_name"]
    update(job, "Reading the uploaded scans", 0.05)
    case, _ = load_job_case(JOBS_DIR / parent.id)
    image = torch.from_numpy(np.stack([case.images[c] for c in MSD_CHANNELS]))
    est = MANIFEST["models"][name].get("cpu_s_per_case_2_threads") or 10
    update(job, f"Running {label} (about {round(est)} s on 2 CPU cores)", 0.15)
    seg = run_model(name, case, image)
    update(job, "Adding it to the 3D view", 0.8)
    add_segmentation(case, JOBS_DIR / parent.id / "bundle", name, seg)
    with jobs_lock:
        parent.models.append(name)
        parent.result["models"][name] = {k: v for k, v in seg.items() if k != "regions"}
        job.result = {"case_url": f"/api/cases/{parent.id}", "upload_id": parent.id, "model": name,
                      "seconds": seg["seconds"]}
    update(job, "Done", 1.0, status="done")


def worker():
    """One job at a time: the models use every CPU core, so running jobs side by side would not be faster."""
    while True:
        job = jobs[work.get()]
        try:
            run_job(job)
        except UploadError as e:  # a problem with the files: tell the user what to fix
            with jobs_lock:
                job.status, job.error = "failed", str(e)
        except Exception as e:  # a bug or a resource problem: log it, give a short message
            traceback.print_exc()
            with jobs_lock:
                job.status, job.error = "failed", f"Something went wrong on the server ({type(e).__name__}). Please try again."


def clean_old_jobs():
    cutoff = time.time() - JOB_TTL_HOURS * 3600
    with jobs_lock:
        old = [j for j in jobs.values() if j.created < cutoff and j.status in ("done", "failed")]
        # a failed upload's scans are of no use: delete them straight away
        old += [j for j in jobs.values() if j.status == "failed" and not j.parent and j not in old]
        for j in old:
            del jobs[j.id]
        known = set(jobs)
    for j in old:
        shutil.rmtree(JOBS_DIR / j.id, ignore_errors=True)
    # folders no job points to (e.g. an upload interrupted before its result was written, which a restart
    # can't reload): removed by their age on disk, so they don't pile up on a long-running server
    if JOBS_DIR.is_dir():
        for d in JOBS_DIR.iterdir():
            if d.is_dir() and d.name not in known and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)


# --- the web app
app = FastAPI(title="Brain tumour segmentation", version="1.0",
              description="Upload 4 MRI scans of a brain; get the tumour segmented by a trained model, as a 3D viewer case.")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])


def restore_jobs():
    """After a restart, pick up finished uploads still on disk (so their links and 'run one more model' keep working)."""
    for job_dir in JOBS_DIR.iterdir():
        meta_file = job_dir / "bundle" / "meta.json"
        if not meta_file.exists() or job_dir.name in jobs:
            continue
        meta = json.loads(meta_file.read_text())
        models = [m for m in meta["segmentations"] if m in MODEL_NAMES]
        jobs[job_dir.name] = Job(
            id=job_dir.name, models=models, status="done", step="Done", progress=1.0,
            created=meta_file.stat().st_mtime,
            result={"case_url": f"/api/cases/{job_dir.name}", "description": meta.get("description"),
                    "models": {m: {k: v for k, v in meta["segmentations"][m].items() if k not in ("zones", "diff", "file")}
                               for m in models}},
        )


@app.on_event("startup")
def start_worker():
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    restore_jobs()
    clean_old_jobs()
    threading.Thread(target=worker, daemon=True).start()


@app.get("/api/health")
def health():
    return {"status": "ok", "models": MODEL_NAMES, "champion": MANIFEST["champion"], "queued": work.qsize()}


@app.get("/api/models")
def models():
    return {
        "registered_model": MANIFEST["registered_model"],
        "champion": MANIFEST["champion"],
        "models": [{"name": n, **{k: v for k, v in MANIFEST["models"][n].items() if k != "onnx_sha256"},
                    "champion": n == MANIFEST["champion"]} for n in MODEL_NAMES],
    }


async def save_upload(upload: UploadFile, folder: Path, name: str, budget: list) -> str:
    """Save one uploaded file, keeping its .nii / .nii.gz ending (the reader needs it); enforce the size limit."""
    fname = upload.filename or ""
    if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
        raise HTTPException(400, f"{fname or 'A file'}: please upload NIfTI files (.nii or .nii.gz).")
    path = folder / (name + (".nii.gz" if fname.endswith(".gz") else ".nii"))
    with open(path, "wb") as f:
        while chunk := await upload.read(1 << 20):
            budget[0] -= len(chunk)
            if budget[0] < 0:
                raise HTTPException(413, f"The upload is larger than {MAX_UPLOAD_MB:.0f} MB.")
            f.write(chunk)
    return str(path)


@app.post("/api/segment", status_code=202)
async def segment(
    scans: UploadFile | None = File(None, description="MSD style: one file with all 4 scans (FLAIR, T1, T1 + contrast, T2)"),
    flair: UploadFile | None = File(None, description="BraTS style: the FLAIR scan"),
    t1: UploadFile | None = File(None, description="BraTS style: the T1 scan"),
    t1ce: UploadFile | None = File(None, description="BraTS style: the T1 + contrast scan"),
    t2: UploadFile | None = File(None, description="BraTS style: the T2 scan"),
    label: UploadFile | None = File(None, description="Optional: the expert outline, to score the models"),
    label_format: str = Form("auto", description="How the outline numbers the zones: auto, msd or brats"),
    models: str = Form("", description="Comma-separated model names; empty = the champion"),
):
    chosen = [m.strip() for m in models.split(",") if m.strip()] or [MANIFEST["champion"]]
    unknown = [m for m in chosen if m not in MODEL_NAMES]
    if unknown:
        raise HTTPException(400, f"Unknown model(s): {', '.join(unknown)}. Available: {', '.join(MODEL_NAMES)}.")
    separate = {"flair": flair, "t1": t1, "t1ce": t1ce, "t2": t2}
    given = [k for k, v in separate.items() if v is not None]
    if scans is not None and given:
        raise HTTPException(400, "Upload either one combined scan file or four separate scan files, not both.")
    if scans is None and len(given) < 4:
        missing = [k for k in separate if k not in given]
        raise HTTPException(400, f"Missing scans: {', '.join(missing)}. Upload one combined file or all four scans.")
    if label_format not in ("auto", "msd", "brats"):
        raise HTTPException(400, "label_format must be auto, msd or brats.")

    clean_old_jobs()
    job = Job(id=secrets.token_urlsafe(9), models=list(dict.fromkeys(chosen)))
    folder = JOBS_DIR / job.id / "input"
    folder.mkdir(parents=True)
    budget = [MAX_UPLOAD_MB * 1e6]
    try:
        saved_scans = (await save_upload(scans, folder, "scans", budget) if scans is not None
                       else {k: await save_upload(v, folder, k, budget) for k, v in separate.items()})
        saved_label = await save_upload(label, folder, "label", budget) if label is not None else None
    except HTTPException:
        shutil.rmtree(JOBS_DIR / job.id, ignore_errors=True)
        raise
    (JOBS_DIR / job.id / "input.json").write_text(json.dumps(
        {"scans": saved_scans, "label": saved_label, "label_format": label_format}))
    with jobs_lock:
        jobs[job.id] = job
    work.put(job.id)
    return {"job_id": job.id, "status_url": f"/api/jobs/{job.id}", "position": work.qsize()}


@app.post("/api/jobs/{job_id}/models", status_code=202)
def add_model(job_id: str, model: str = Form(..., description="The model to run on this upload")):
    upload = jobs.get(job_id)
    if upload is None or upload.parent:
        raise HTTPException(404, "This upload is no longer on the server (uploads are kept for a few hours). Please upload it again.")
    if not (JOBS_DIR / job_id / "input").is_dir():
        raise HTTPException(410, "The scans of this upload are no longer on the server, so no more models can be run on it. "
                                 "Please upload them again.")
    if upload.status != "done":
        raise HTTPException(409, "That upload is not finished yet.")
    if model not in MODEL_NAMES:
        raise HTTPException(400, f"Unknown model: {model}. Available: {', '.join(MODEL_NAMES)}.")
    if model in upload.models:
        raise HTTPException(409, f"{MANIFEST['models'][model]['display_name']} has already been run on this upload.")
    job = Job(id=secrets.token_urlsafe(9), models=[model], parent=job_id)
    with jobs_lock:
        jobs[job.id] = job
    work.put(job.id)
    return {"job_id": job.id, "status_url": f"/api/jobs/{job.id}", "position": work.qsize()}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job (results are kept for a few hours).")
    with jobs_lock:
        out = asdict(job)
    if job.status == "queued":  # how many jobs are ahead of this one
        out["position"] = sum(1 for j in jobs.values() if j.status == "queued" and j.created < job.created) + 1
    return out


@app.get("/api/cases/{job_id}/{file_path:path}")
def case_file(job_id: str, file_path: str):
    bundle = (JOBS_DIR / job_id / "bundle").resolve()
    path = (bundle / file_path).resolve()
    if bundle not in path.parents or not path.is_file():  # no escaping the bundle folder with ../
        raise HTTPException(404, "Not found.")
    return FileResponse(path)


# the built viewer (and its gallery), when present: the whole app from one server
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="viewer")
