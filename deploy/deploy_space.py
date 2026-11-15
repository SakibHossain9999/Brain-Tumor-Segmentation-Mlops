"""Deploy to a Hugging Face Space (Docker SDK): upload what the Dockerfile needs; the Space builds and runs it.

    HF_TOKEN=... python deploy/deploy_space.py --space USER/brain-tumor-segmentation [--dry-run]

Called by GitHub Actions after all checks pass on main. The Space builds the same Dockerfile CI tested,
which downloads the models and data from the Hugging Face model repo.
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
# only what the image is built from (see Dockerfile); everything else stays on GitHub
INCLUDE = [
    "Dockerfile", ".dockerignore", "requirements/api.txt",
    "api/__init__.py", "api/main.py", "api/uploads.py",
    "ml/*.py", "pipeline/pyproject.toml", "pipeline/brainviz/*.py",
    "splits/*.json", "models/*.json", "scripts/hub.py",
    "web/*.html", "web/package.json", "web/package-lock.json", "web/tsconfig.json", "web/vite.config.js",
    "web/src/*.ts", "web/public/data/*.json",
]


def stage(dst: Path) -> list[str]:
    """Copy the files to deploy into dst, plus the Space's README (its settings page)."""
    files = sorted({p for pat in INCLUDE for p in ROOT.glob(pat) if p.is_file()})
    for p in files:
        (dst / p.relative_to(ROOT)).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst / p.relative_to(ROOT))
    shutil.copy2(ROOT / "deploy" / "space_README.md", dst / "README.md")
    return [str(p.relative_to(ROOT)) for p in files] + ["README.md"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="USER/space-name")
    ap.add_argument("--message", default=os.environ.get("GITHUB_SHA", "manual")[:7])
    ap.add_argument("--dry-run", action="store_true", help="only list what would be uploaded")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        names = stage(Path(tmp))
        size = sum(f.stat().st_size for f in Path(tmp).rglob("*") if f.is_file())
        print(f"{len(names)} files, {size / 1e6:.1f} MB")
        if a.dry_run:
            print("\n".join(names))
            return
        api = HfApi()  # token from HF_TOKEN or `hf auth login`
        api.create_repo(a.space, repo_type="space", space_sdk="docker", exist_ok=True)
        # one commit: replace everything in the Space with exactly these files
        api.upload_folder(folder_path=tmp, repo_id=a.space, repo_type="space", delete_patterns="*",
                          commit_message=f"Deploy {a.message}")
    print(f"deployed: https://huggingface.co/spaces/{a.space} (the Space now builds the image, ~10 min)")


if __name__ == "__main__":
    main()
