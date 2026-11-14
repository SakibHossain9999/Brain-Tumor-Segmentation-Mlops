"""Put the exported ONNX models in the MLflow model registry and choose the champion.

    python ml/register.py --champion segresnet

Registry layout: one registered model "brain-tumour-seg" with one version per architecture.
Each version is the ONNX file, logged inside the training run that produced it, so any
deployed model can be traced back to its config, curves, git commit and data split.
Aliases point at versions:
    @champion                              the default model of the app
    @segresnet, @swin_unetr, @segformer3d  each model by name
Promoting another model later = moving @champion (re-run with --champion <name>); no code changes.

The deployed app has no access to this MLflow database, so the result is also written to
models/manifest.json, which the API reads.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import onnx
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.evaluate import MODELS, NICE  # noqa: E402

REGISTERED = "brain-tumour-seg"
MODELS_DIR = Path("models")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def training_run(client, experiment_id, name):
    """The finished training run of one model (not its test run)."""
    runs = client.search_runs([experiment_id], f"attributes.run_name = '{name}' and attributes.status = 'FINISHED'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise SystemExit(f"no finished training run called {name!r}")
    return runs[0]


def existing_version(client, name, digest):
    """A version already registered for this exact ONNX file (so re-running doesn't duplicate)."""
    for v in client.search_model_versions(f"name = '{REGISTERED}'"):
        if v.tags.get("architecture") == name and v.tags.get("onnx_sha256") == digest:
            return v
    return None


def register(client, experiment_id, name):
    path = MODELS_DIR / f"{name}.onnx"
    meta = json.loads(path.with_suffix(".json").read_text())
    digest = sha256(path)
    version = existing_version(client, name, digest)
    if version is None:
        run = training_run(client, experiment_id, name)
        shape = meta["input"]["shape"]
        signature = ModelSignature(  # what goes in and comes out: documents the model's contract
            inputs=Schema([TensorSpec(np.dtype(np.float32), tuple(shape), "scans")]),
            outputs=Schema([TensorSpec(np.dtype(np.float32), (shape[0], 3, *shape[2:]), "logits")]),
        )
        with mlflow.start_run(run_id=run.info.run_id):  # reopen the training run: the model joins its history
            info = mlflow.onnx.log_model(onnx.load(str(path)), name="onnx_model", signature=signature,
                                         registered_model_name=REGISTERED)
            mlflow.log_artifact(str(path.with_suffix(".json")), artifact_path="onnx_model_meta")
        version = client.get_model_version(REGISTERED, str(info.registered_model_version))
        print(f"  registered {name} as version {version.version} (training run {run.info.run_id[:8]})")
    else:
        print(f"  {name}: this exact file is already version {version.version}")

    tags = {
        "architecture": name,
        "onnx_sha256": digest,
        "checkpoint_step": meta["checkpoint_step"],
        "val_dice_mean": meta["val_dice_mean"],
        "test_dice_mean": meta["test_dice_mean"],
        "onnx_vs_pytorch_dice": meta.get("onnx_vs_pytorch_dice"),
        "cpu_s_per_case_2_threads": meta.get("cpu_s_per_case_2_threads"),
    }
    for k, v in tags.items():
        client.set_model_version_tag(REGISTERED, version.version, k, str(v))
    client.update_model_version(REGISTERED, version.version,
                                description=f"{NICE[name]}, best validation checkpoint (step {meta['checkpoint_step']}). "
                                            f"Test Dice {meta['test_dice_mean']}.")
    client.set_registered_model_alias(REGISTERED, name, version.version)
    return version, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", required=True, choices=MODELS, help="the app's default model")
    ap.add_argument("--reason", default="", help="why this model is the champion (stored in the registry)")
    args = ap.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_id = mlflow.set_experiment("brain-npr").experiment_id
    client = mlflow.MlflowClient()
    try:
        client.get_registered_model(REGISTERED)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(REGISTERED, description="Brain tumour segmentation (MSD Task01): "
                                       "4 MRI scans in, whole tumour / tumour core / active tumour out.")

    manifest = {"registered_model": REGISTERED, "champion": args.champion, "models": {}}
    for name in MODELS:
        print(f"=== {name}")
        version, meta = register(client, experiment_id, name)
        manifest["models"][name] = {
            "display_name": NICE[name],
            "version": int(version.version),
            "onnx": f"{name}.onnx",
            "onnx_sha256": version.tags.get("onnx_sha256") or sha256(MODELS_DIR / f"{name}.onnx"),
            "test_dice_mean": meta["test_dice_mean"],
            "cpu_s_per_case_2_threads": meta.get("cpu_s_per_case_2_threads"),
        }

    champion = manifest["models"][args.champion]["version"]
    client.set_registered_model_alias(REGISTERED, "champion", champion)
    if args.reason:
        client.set_model_version_tag(REGISTERED, champion, "champion_reason", args.reason)
    manifest["champion_reason"] = args.reason
    (MODELS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n@champion -> {args.champion} (version {champion}); wrote {MODELS_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
