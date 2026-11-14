"""Large files on the Hugging Face Hub: the ONNX models, the sample brains and the gallery cases.

GitHub refuses files over 100 MB (swin_unetr.onnx is 260 MB) and the gallery is ~100 MB of 3D files,
so they live in a Hugging Face model repository instead; Docker, CI and the website download them.

    python scripts/hub.py upload   --repo USER/brain-tumor-segmentation-mlops   # once logged in: hf auth login
    python scripts/hub.py download --repo USER/brain-tumor-segmentation-mlops --what models samples cases --to .

Layout on the Hub (and where `download` puts it, relative to --to):
    models/<name>.onnx, models/<name>.json, models/manifest.json   -> models/
    samples/*                                                    -> output/samples/  (or --samples-dir)
    cases/*                                                      -> web/public/cases/ (or --cases-dir)
"""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

MODEL_CARD = """---
license: cc-by-sa-4.0
tags: [medical-imaging, brain-tumor-segmentation, brats, onnx, monai, segresnet, swin-unetr, segformer3d]
---
# Brain tumor segmentation models (ONNX) + demo data

Files for [Brain-Tumor-Segmentation-Mlops](https://github.com/SakibHossain9999/Brain-Tumor-Segmentation-Mlops):

| Folder | Contents |
|---|---|
| `models/` | SegResNet (champion), Swin UNETR, SegFormer3D as ONNX (one 160³ window in, WT/TC/ET logits out), each with its training config and test results; `manifest.json` = which one is the champion |
| `samples/` | 4 test-set brains (4 MRI scans in one file + expert outline) to try the upload |
| `cases/` | the 3D viewer's sample gallery (meshes + compressed scans) |

Trained on the Medical Segmentation Decathlon Task01_BrainTumour (from BraTS 2016/2017), CC-BY-SA 4.0.
Test-set mean Dice (72 brains): SegResNet 0.835, Swin UNETR 0.835, SegFormer3D 0.813.
Research and education only: not a medical device.
"""


def upload(repo: str, root: Path, samples_dir: Path, cases_dir: Path):
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    (root / "README.hub.md").write_text(MODEL_CARD)
    try:
        api.upload_file(path_or_fileobj=root / "README.hub.md", path_in_repo="README.md", repo_id=repo,
                        commit_message="Model card")
    finally:
        (root / "README.hub.md").unlink()
    api.upload_folder(folder_path=root / "models", path_in_repo="models", repo_id=repo,
                      allow_patterns=["*.onnx", "*.json"], commit_message="ONNX models + manifest")
    api.upload_folder(folder_path=samples_dir, path_in_repo="samples", repo_id=repo, commit_message="Sample brains")
    api.upload_folder(folder_path=cases_dir, path_in_repo="cases", repo_id=repo, commit_message="Gallery cases")
    print(f"uploaded to https://huggingface.co/{repo}")


def download(repo: str, what: list[str], root: Path, samples_dir: Path, cases_dir: Path):
    patterns = {"models": "models/*", "samples": "samples/*", "cases": "cases/**"}
    cache = Path(snapshot_download(repo, allow_patterns=[patterns[w] for w in what]))
    targets = {"models": root / "models", "samples": samples_dir, "cases": cases_dir}
    for w in what:
        src, dst = cache / w, targets[w]
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        n = sum(1 for f in dst.rglob("*") if f.is_file())
        print(f"{w}: {n} files -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["upload", "download"])
    ap.add_argument("--repo", required=True, help="Hugging Face repo, e.g. USER/brain-tumor-segmentation-mlops")
    ap.add_argument("--to", type=Path, default=Path("."), help="project root")
    ap.add_argument("--what", nargs="+", default=["models", "samples", "cases"], choices=["models", "samples", "cases"])
    ap.add_argument("--samples-dir", type=Path, help="default: <to>/output/samples")
    ap.add_argument("--cases-dir", type=Path, help="default: <to>/web/public/cases")
    a = ap.parse_args()
    samples = a.samples_dir or a.to / "output" / "samples"
    cases = a.cases_dir or a.to / "web" / "public" / "cases"
    if a.action == "upload":
        upload(a.repo, a.to, samples, cases)
    else:
        download(a.repo, a.what, a.to, samples, cases)


if __name__ == "__main__":
    main()
