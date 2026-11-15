"""Back up / restore the training files that exist on the training machine only, to a PRIVATE Hugging Face repo.

    python scripts/backup.py backup  --repo USER/brain-tumor-segmentation-training
    python scripts/backup.py restore --repo USER/brain-tumor-segmentation-training --to .

What is backed up (everything else is in git, on the public Hub repo, or downloadable):
    checkpoints/<model>/best.pt   PyTorch weights of the best checkpoint (output/runs/<model>/best.pt)
    mlflow/mlflow.db              MLflow tracking + model registry database (consistent snapshot)
    mlflow/mlruns.tar             MLflow run files: curves, images, logged and registered models
    preds/<model>/*.nii.gz        test-set predictions (output/preds)
    logs/*.log                    training / evaluation logs (output/logs)
    backup.json                   where it came from, when, and checksums

MLflow stores absolute folder paths in its database; `restore` rewrites them to the new location,
so `mlflow ui --backend-store-uri sqlite:///mlflow.db` works there straight away.
"""
import argparse
import hashlib
import json
import shutil
import sqlite3
import tarfile
import tempfile
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download

MODELS = ["segresnet", "swin_unetr", "segformer3d"]

README = """---
license: other
tags: [backup]
---
# Training backup (private)

Checkpoints, MLflow history, test predictions and logs for
[Brain-Tumor-Segmentation-Mlops](https://github.com/SakibHossain9999/Brain-Tumor-Segmentation-Mlops).
Restore into a clone of that repository with:

    python scripts/backup.py restore --repo {repo} --to .
    mlflow ui --backend-store-uri sqlite:///mlflow.db

Contents and checksums: `backup.json`.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def backup(repo: str, root: Path):
    root = root.resolve()
    api = HfApi()
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    ops, info = [], {"source_root": str(root), "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "files": {}}

    def add(local: Path, remote: str):
        ops.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)))
        info["files"][remote] = {"bytes": local.stat().st_size, "sha256": sha256(local)}

    with tempfile.TemporaryDirectory(dir=root / "output") as tmp:
        tmp = Path(tmp)
        for m in MODELS:
            add(root / "output" / "runs" / m / "best.pt", f"checkpoints/{m}/best.pt")
        # MLflow database: copy through SQLite's backup API, so it is consistent even if MLflow is open
        src, dst = sqlite3.connect(root / "mlflow.db"), sqlite3.connect(tmp / "mlflow.db")
        src.backup(dst)
        src.close(), dst.close()
        add(tmp / "mlflow.db", "mlflow/mlflow.db")
        with tarfile.open(tmp / "mlruns.tar", "w") as t:  # 300+ small files -> one file (ONNX/weights don't compress)
            t.add(root / "mlruns", arcname="mlruns")
        add(tmp / "mlruns.tar", "mlflow/mlruns.tar")
        for folder, remote in [("output/preds", "preds"), ("output/logs", "logs")]:
            for f in sorted((root / folder).rglob("*")):
                if f.is_file():
                    add(f, f"{remote}/{f.relative_to(root / folder).as_posix()}")
        (tmp / "backup.json").write_text(json.dumps(info, indent=1))
        ops.append(CommitOperationAdd(path_in_repo="backup.json", path_or_fileobj=str(tmp / "backup.json")))
        (tmp / "README.md").write_text(README.format(repo=repo))
        ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(tmp / "README.md")))
        total = sum(v["bytes"] for v in info["files"].values())
        print(f"uploading {len(ops)} files, {total / 1e6:.0f} MB ...")
        api.create_commit(repo_id=repo, operations=ops, commit_message=f"Training backup {info['created']}")
    print(f"backed up to https://huggingface.co/{repo} (private)")


def fix_mlflow_paths(db: Path, old: str, new: str) -> int:
    """Replace the old project folder with the new one in every text column of the MLflow database."""
    con, changed = sqlite3.connect(db), 0
    tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
    for t in tables:
        for col in [r[1] for r in con.execute(f'pragma table_info("{t}")')]:
            try:
                cur = con.execute(f'update "{t}" set "{col}" = replace("{col}", ?, ?) where "{col}" like ?',
                                  (old, new, f"%{old}%"))
                changed += cur.rowcount
            except sqlite3.Error:
                pass  # non-text column
    con.commit()
    con.close()
    return changed


def restore(repo: str, root: Path):
    root = root.resolve()
    cache = Path(snapshot_download(repo))
    info = json.loads((cache / "backup.json").read_text())
    for remote, meta in info["files"].items():  # check nothing was corrupted on the way
        if sha256(cache / remote) != meta["sha256"]:
            raise SystemExit(f"checksum mismatch: {remote}")
    for m in MODELS:
        (root / "output" / "runs" / m).mkdir(parents=True, exist_ok=True)
        shutil.copy2(cache / "checkpoints" / m / "best.pt", root / "output" / "runs" / m / "best.pt")
    for remote, local in [("preds", "output/preds"), ("logs", "output/logs")]:
        if (cache / remote).exists():
            shutil.copytree(cache / remote, root / local, dirs_exist_ok=True)
    if (root / "mlflow.db").exists() or (root / "mlruns").exists():
        raise SystemExit("mlflow.db or mlruns/ already exists here: move them away first")
    shutil.copy2(cache / "mlflow" / "mlflow.db", root / "mlflow.db")
    with tarfile.open(cache / "mlflow" / "mlruns.tar") as t:
        t.extractall(root, filter="data")
    n = fix_mlflow_paths(root / "mlflow.db", info["source_root"], str(root))
    print(f"restored into {root} (backup from {info['created']}); {n} MLflow paths updated")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["backup", "restore"])
    ap.add_argument("--repo", required=True, help="private Hugging Face repo, e.g. USER/brain-tumor-segmentation-training")
    ap.add_argument("--to", type=Path, default=Path("."), help="project root")
    a = ap.parse_args()
    (backup if a.action == "backup" else restore)(a.repo, a.to)


if __name__ == "__main__":
    main()
