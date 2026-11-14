"""End-to-end check of a running upload service, as a client would use it.

    python api/test_api.py [http://localhost:8000]

Needs the sample brains in output/samples (scripts/make_samples.py).
"""
import sys
import tempfile
import time
from pathlib import Path

import httpx
import nibabel as nib
import numpy as np

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
S = Path("output/samples")
client = httpx.Client(base_url=URL, timeout=120)
failures = []


def expect(ok: bool, what: str):
    print(("  pass  " if ok else "  FAIL  ") + what)
    if not ok:
        failures.append(what)


def run(files: dict, data: dict | None = None):
    """Upload, then poll the job until it finishes. Returns (http status, job json or error body)."""
    handles = {k: (Path(p).name, open(p, "rb")) for k, p in files.items()}
    r = client.post("/api/segment", files=handles, data=data or {})
    if r.status_code != 202:
        return r.status_code, r.json()
    job_id = r.json()["job_id"]
    t0, last = time.time(), ""
    while True:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["step"] != last:
            print(f"    {time.time() - t0:5.1f} s  {job['progress']:.0%}  {job['step']}")
            last = job["step"]
        if job["status"] in ("done", "failed"):
            return 202, job
        time.sleep(0.5)


print("1. MSD-style file + expert outline, two models")
code, job = run({"scans": S / "BRATS_075.nii.gz", "label": S / "BRATS_075_seg.nii.gz"},
                {"models": "segresnet,segformer3d"})
expect(job["status"] == "done", "job finished")
res = job["result"]
expect(res["identified"]["case_id"] == "BRATS_075" and res["identified"]["split"] == "test", "recognised as test brain BRATS_075")
expect(abs(res["models"]["segresnet"]["dice_mean"] - 0.876) < 0.02, f"SegResNet Dice here {res['models']['segresnet']['dice_mean']} (evaluation said ~0.876)")
meta = client.get(f"{res['case_url']}/meta.json").json()
expect(list(meta["segmentations"]) == ["expert", "segresnet", "segformer3d"], "bundle has expert + both models")
expect(meta["default_segmentation"] == "segresnet", "champion shown first")
expect(client.get(f"{res['case_url']}/brain.glb").status_code == 200, "bundle files are served")
expect(client.get(f"{res['case_url']}/../../models/manifest.json").status_code == 404, "cannot escape the bundle folder")

print("2. BraTS-style: 4 separate files, no outline, default model")
img = nib.load(S / "BRATS_238.nii.gz")
data = np.asarray(img.dataobj)
tmp = Path(tempfile.mkdtemp())
four = {}
for i, slot in enumerate(["flair", "t1", "t1ce", "t2"]):
    four[slot] = tmp / f"case_{slot}.nii.gz"
    nib.save(nib.Nifti1Image(data[..., i], img.affine), four[slot])
code, job = run(four)
expect(job["status"] == "done", "job finished")
expect(list(job["result"]["models"]) == ["segresnet"], "default model = champion")
expect("scores" not in job["result"]["models"]["segresnet"], "no scores without an outline")

print("2b. Run one more model on that upload afterwards")
upload_id = job["id"]
r = client.post(f"/api/jobs/{upload_id}/models", data={"model": "segformer3d"})
expect(r.status_code == 202, f"accepted ({r.status_code})")
while (extra := client.get(f"/api/jobs/{r.json()['job_id']}").json())["status"] not in ("done", "failed"):
    time.sleep(0.5)
expect(extra["status"] == "done", f"extra model finished ({extra.get('error')})")
meta = client.get(f"/api/cases/{upload_id}/meta.json").json()
expect(list(meta["segmentations"]) == ["segresnet", "segformer3d"], f"bundle now has both: {list(meta['segmentations'])}")
expect(client.get(f"/api/cases/{upload_id}/seg_segformer3d.glb").status_code == 200, "its 3D file is served")
again = client.post(f"/api/jobs/{upload_id}/models", data={"model": "segformer3d"})
expect(again.status_code == 409, f"running it twice is refused: {again.json().get('detail')}")

print("3. Mistakes")
code, body = run({"scans": S / "BRATS_075.nii.gz"}, {"models": "unet9000"})
expect(code == 400 and "Unknown model" in body["detail"], f"unknown model -> 400: {body.get('detail')}")
code, body = run({"flair": four["flair"], "t1": four["t1"]})
expect(code == 400 and "Missing scans" in body["detail"], f"missing scans -> 400: {body.get('detail')}")
(tmp / "notes.txt").write_text("hello")
code, body = run({"scans": tmp / "notes.txt"})
expect(code == 400 and "NIfTI" in body["detail"], f"wrong file type -> 400: {body.get('detail')}")
head = data.copy()
head[head == 0] = np.random.default_rng(0).uniform(1, 20, (head == 0).sum())
nib.save(nib.Nifti1Image(head, img.affine), tmp / "head.nii.gz")
code, job = run({"scans": tmp / "head.nii.gz"})
expect(job["status"] == "failed" and "skull-stripped" in job["error"], f"whole-head scan -> job fails: {job.get('error')}")

print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)
