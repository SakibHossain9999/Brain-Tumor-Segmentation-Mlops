"""Convert the trained models to ONNX and check the ONNX versions give the same answers.

    python ml/export_onnx.py                       # all three
    python ml/export_onnx.py --models segresnet

ONNX is a portable model format: the API runs it with onnxruntime on a plain CPU, without
PyTorch's training code or a GPU. For each model this writes
    models/<name>.onnx    the network (one 160^3 window in, WT/TC/ET scores out)
    models/<name>.json    its training config + test results, needed to use it
and then checks, on real test brains, that ONNX on the CPU and PyTorch on the GPU agree.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.data import get_cases  # noqa: E402
from ml.inference import OnnxModel, load_model, predict  # noqa: E402

MODELS = ["segformer3d", "segresnet", "swin_unetr"]
OUT = Path("models")


def export(name):
    model, cfg, ckpt = load_model(f"output/runs/{name}/best.pt", "cpu")  # export in full precision on the CPU
    crop = cfg["data"]["crop_size"]
    window = torch.randn(1, 4, *crop)
    path = OUT / f"{name}.onnx"
    t0 = time.perf_counter()
    try:  # the newer exporter (reads the Python code)
        torch.onnx.export(model, (window,), path, input_names=["scans"], output_names=["logits"],
                          opset_version=18, dynamo=True, external_data=False)
        how = "dynamo"
    except Exception as e:  # the older exporter (traces one run) handles some layers the new one can't
        print(f"  new exporter failed ({type(e).__name__}: {str(e)[:200]}), trying the older one")
        torch.onnx.export(model, (window,), path, input_names=["scans"], output_names=["logits"],
                          opset_version=18, dynamo=False)
        how = "trace"
    print(f"  exported with the {how} exporter in {time.perf_counter() - t0:.0f} s -> {path} ({path.stat().st_size / 1e6:.1f} MB)")

    # does one window give the same scores?
    with torch.no_grad():
        ref = model(window).numpy()
    got = OnnxModel(path)(window).numpy()
    print(f"  one random window: largest difference in scores {np.abs(ref - got).max():.2e} (scores range about +-{np.abs(ref).max():.0f})")

    summary_file = Path(f"results/{name}/test_summary.json")
    test = json.loads(summary_file.read_text()) if summary_file.exists() else {}
    path.with_suffix(".json").write_text(json.dumps({
        "model": name,
        "input": {"name": "scans", "shape": [1, 4, *crop], "channels": ["FLAIR", "T1", "T1 + contrast", "T2"]},
        "output": {"name": "logits", "regions": ["WT", "TC", "ET"], "threshold": 0.5, "activation": "sigmoid"},
        "checkpoint_step": ckpt["step"],
        "val_dice_mean": round(ckpt["val_dice_mean"], 4),
        "test_dice_mean": test.get("dice_mean"),
        "exporter": how,
        "config": cfg,
    }, indent=2))
    return model, cfg


def check_full_brains(name, model, cfg, n_cases, threads):
    """Whole test brains: PyTorch on the GPU (as evaluated) vs ONNX on the CPU (as the API will run it)."""
    model = model.to("cuda")
    onnx_model = OnnxModel(OUT / f"{name}.onnx", threads)
    load = Compose([LoadImaged(keys="image"), EnsureChannelFirstd(keys="image")])
    agree, times = [], []
    for case in get_cases("test", n_cases):
        image = load(case)["image"].as_tensor()
        a = predict(model, image, cfg).cpu() > 0.5
        t0 = time.perf_counter()
        b = predict(onnx_model, image, cfg) > 0.5
        times.append(time.perf_counter() - t0)
        dice = [float(2 * (a[r] & b[r]).sum() / max(int(a[r].sum() + b[r].sum()), 1)) for r in range(3)]
        agree.append(dice)
        print(f"  {case['case_id']}: GPU vs ONNX masks agree, Dice WT {dice[0]:.4f} TC {dice[1]:.4f} ET {dice[2]:.4f}"
              f"   ONNX on {threads} CPU threads: {times[-1]:.1f} s")
    return {"onnx_vs_pytorch_dice": round(float(np.mean(agree)), 4),
            f"cpu_s_per_case_{threads}_threads": round(float(np.mean(times)), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--check-cases", type=int, default=3, help="test brains to compare GPU vs ONNX on")
    ap.add_argument("--threads", type=int, nargs="+", default=[2, 8],
                    help="CPU threads to time ONNX with (2 = a free Hugging Face Space)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for name in args.models:
        print(f"\n=== {name}")
        model, cfg = export(name)
        meta_file = OUT / f"{name}.json"
        meta = json.loads(meta_file.read_text())
        for threads in args.threads:
            meta.update(check_full_brains(name, model, cfg, args.check_cases, threads))
        meta_file.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
