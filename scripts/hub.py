"""Large files on the Hugging Face Hub: the ONNX models, the sample brains and the gallery cases.

GitHub refuses files over 100 MB (swin_unetr.onnx is 260 MB) and the gallery is ~100 MB of 3D files,
so they live in a Hugging Face model repository instead; Docker, CI and the website download them.

    python scripts/hub.py upload   --repo USER/brain-tumor-segmentation-mlops   # once logged in: hf auth login
    python scripts/hub.py card     --repo USER/brain-tumor-segmentation-mlops   # only the Hub page (scripts/hub_model_card.md)
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

MODEL_CARD = Path(__file__).with_name("hub_model_card.md")  # the Hub page (README.md there)


def upload(repo: str, root: Path, samples_dir: Path, cases_dir: Path):
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    upload_card(repo)
    api.upload_folder(folder_path=root / "models", path_in_repo="models", repo_id=repo,
                      allow_patterns=["*.onnx", "*.json"], commit_message="ONNX models + manifest")
    api.upload_folder(folder_path=samples_dir, path_in_repo="samples", repo_id=repo, commit_message="Sample brains")
    api.upload_folder(folder_path=cases_dir, path_in_repo="cases", repo_id=repo, commit_message="Gallery cases")
    print(f"uploaded to https://huggingface.co/{repo}")


def upload_card(repo: str):
    HfApi().upload_file(path_or_fileobj=MODEL_CARD, path_in_repo="README.md", repo_id=repo, commit_message="Model card")
    print(f"model card updated: https://huggingface.co/{repo}")


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
    ap.add_argument("action", choices=["upload", "card", "download"])
    ap.add_argument("--repo", required=True, help="Hugging Face repo, e.g. USER/brain-tumor-segmentation-mlops")
    ap.add_argument("--to", type=Path, default=Path("."), help="project root")
    ap.add_argument("--what", nargs="+", default=["models", "samples", "cases"], choices=["models", "samples", "cases"])
    ap.add_argument("--samples-dir", type=Path, help="default: <to>/output/samples")
    ap.add_argument("--cases-dir", type=Path, help="default: <to>/web/public/cases")
    a = ap.parse_args()
    samples = a.samples_dir or a.to / "output" / "samples"
    cases = a.cases_dir or a.to / "web" / "public" / "cases"
    if a.action == "card":
        upload_card(a.repo)
    elif a.action == "upload":
        upload(a.repo, a.to, samples, cases)
    else:
        download(a.repo, a.what, a.to, samples, cases)


if __name__ == "__main__":
    main()
