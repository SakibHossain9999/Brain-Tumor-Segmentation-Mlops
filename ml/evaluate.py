"""Score trained models on the test set (72 brains none of them has seen).

    python ml/evaluate.py                                   # all three models, test set
    python ml/evaluate.py --models segresnet --split val --limit 10   # quick check

Per model -> results/<model>/
    test_per_case.csv   every case x region: Dice, HD95, sensitivity, precision, true and predicted volume
    test_summary.json   averages + efficiency (parameters, compute, prediction time, GPU memory, training speed)
All models -> results/
    comparison.md / comparison.json   the side-by-side table
    test_dice_boxplot.png             spread of Dice over cases, per region and model
Also logged to MLflow as runs "test-<model>". With --save-preds, predicted label maps go to output/preds/<model>/.
"""
import argparse
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path

import matplotlib
import mlflow
import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.data import REGIONS, ConvertMSDToRegionsd, get_cases  # noqa: E402
from ml.inference import load_model, predict, regions_to_labels  # noqa: E402
from ml.metrics import region_metrics  # noqa: E402

MODELS = ["segformer3d", "segresnet", "swin_unetr"]
NICE = {"segformer3d": "SegFormer3D", "segresnet": "SegResNet", "swin_unetr": "Swin UNETR"}
# the same colour per model as on the web Models page (checked for colour-blind safety)
COLORS = {"segresnet": "#2a78d6", "swin_unetr": "#eb6834", "segformer3d": "#1baf7a"}
METRICS = ["dice", "hd95", "sensitivity", "precision"]
# ---------------------------------------------------------------- efficiency numbers
def count_flops(model, cfg, device):
    """Multiply-adds for ONE 160^3 window, in GFLOPs (a hardware-independent measure of compute)."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        x = torch.zeros(1, 4, *cfg["data"]["crop_size"], device=device)
        counter = FlopCounterMode(display=False)
        with counter, torch.no_grad():
            model(x)
        return round(counter.get_total_flops() / 1e9, 1)
    except Exception as e:  # not essential: some layers can't be counted
        print(f"  (FLOP count skipped: {e})")
        return None


def training_speed(model_name):
    """Median seconds per training step, read from the training log (validation time excluded)."""
    log = Path(f"output/logs/{model_name}.log")
    rates = [float(m) for m in re.findall(r"([\d.]+) s/step", log.read_text())] if log.exists() else []
    return round(statistics.median(rates), 3) if rates else None


def summarise(rows):
    """Per-case rows -> mean / median / std of every metric, per region."""
    out = {}
    for region in REGIONS:
        for metric in METRICS:
            vals = np.array([r[metric] for r in rows if r["region"] == region], dtype=float)
            vals = vals[~np.isnan(vals)]
            out[f"{metric}_{region.lower()}"] = {
                "mean": round(float(vals.mean()), 4), "median": round(float(np.median(vals)), 4),
                "std": round(float(vals.std()), 4), "n": int(vals.size),
            }
    out["dice_mean"] = round(float(np.mean([out[f"dice_{r.lower()}"]["mean"] for r in REGIONS])), 4)
    return out


# ---------------------------------------------------------------- one model
def evaluate_model(name, cases, args, device):
    model, cfg, ckpt = load_model(f"output/runs/{name}/best.pt", device)
    print(f"\n=== {NICE[name]}: checkpoint from step {ckpt['step']} (validation Dice {ckpt['val_dice_mean']:.4f})")

    loader = DataLoader(
        Dataset(cases, Compose([LoadImaged(keys=["image", "label"]), EnsureChannelFirstd(keys=["image", "label"]),
                                ConvertMSDToRegionsd(keys="label")])),
        batch_size=1, num_workers=4,  # next cases load from disk while the GPU works
    )
    pred_dir = Path(f"output/preds/{name}")
    if args.save_preds:
        pred_dir.mkdir(parents=True, exist_ok=True)

    rows, times = [], []
    torch.cuda.reset_peak_memory_stats()
    for i, batch in enumerate(loader):
        case_id = batch["case_id"][0]
        image, label = batch["image"][0], batch["label"][0]
        spacing = [float(s) for s in image.pixdim[:3]]
        voxel_ml = float(np.prod(spacing)) / 1000

        torch.cuda.synchronize(); t0 = time.perf_counter()
        probs = predict(model, image.as_tensor(), cfg, overlap=args.overlap)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)

        pred = probs > 0.5
        true = label.as_tensor().to(device).bool()
        for r, region in enumerate(REGIONS):
            m = region_metrics(pred[r], true[r], spacing)
            rows.append({"case_id": case_id, "region": region, **{k: round(v, 4) for k, v in m.items()},
                         "true_ml": round(int(true[r].sum()) * voxel_ml, 2),
                         "pred_ml": round(int(pred[r].sum()) * voxel_ml, 2)})
        if args.save_preds:
            nib.save(nib.Nifti1Image(regions_to_labels(pred), np.asarray(image.affine)), pred_dir / f"{case_id}.nii.gz")
        dice = [r["dice"] for r in rows[-3:]]
        print(f"  [{i + 1:2d}/{len(cases)}] {case_id}  Dice WT {dice[0]:.3f}  TC {dice[1]:.3f}  ET {dice[2]:.3f}  ({times[-1]:.2f} s)", flush=True)

    summary = summarise(rows)
    summary["efficiency"] = {
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "gflops_per_window": count_flops(model, cfg, device),
        # the first case includes one-off GPU warm-up, so it is left out of the timing
        "predict_s_per_case": round(statistics.mean(times[1:] or times), 3),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "train_s_per_step": training_speed(name),
        "train_note": "gradient checkpointing on (same results, ~25% slower)" if cfg["model"].get("use_checkpoint") else "",
    }
    summary.update({"model": name, "split": args.split, "n_cases": len(cases), "checkpoint_step": ckpt["step"],
                    "val_dice_mean": round(ckpt["val_dice_mean"], 4), "overlap": args.overlap})

    out = Path(f"results/{name}")
    out.mkdir(parents=True, exist_ok=True)
    tag = "" if args.split == "test" else f"_{args.split}"
    with open(out / f"test_per_case{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / f"test_summary{tag}.json").write_text(json.dumps(summary, indent=2))

    if args.split == "test" and not args.limit:  # only full test runs go to MLflow
        with mlflow.start_run(run_name=f"test-{name}"):
            mlflow.set_tags({"stage": "test", "model": name})
            mlflow.log_params({"checkpoint_step": ckpt["step"], "n_cases": len(cases), "overlap": args.overlap})
            mlflow.log_metrics({k: v["mean"] for k, v in summary.items() if isinstance(v, dict) and "mean" in v})
            mlflow.log_metric("dice_mean", summary["dice_mean"])
            mlflow.log_metrics({k: v for k, v in summary["efficiency"].items() if isinstance(v, (int, float))})
            mlflow.log_artifacts(str(out), artifact_path="results")
    del model
    torch.cuda.empty_cache()
    return summary, rows


# ---------------------------------------------------------------- all models side by side
def compare(results):
    names = list(results)
    lines = [
        "| Model | Dice WT | Dice TC | Dice ET | **Dice mean** | HD95 WT | HD95 TC | HD95 ET | Sens. ET | Prec. ET "
        "| Params (M) | GFLOPs / window | Predict (s/case) | GPU (GB) | Train (s/step) |",
        "|" + "---|" * 15,
    ]
    for n in names:
        s, e = results[n][0], results[n][0]["efficiency"]
        g = lambda k: s[k]["mean"]  # noqa: E731
        lines.append(
            f"| {NICE[n]} | {g('dice_wt'):.3f} | {g('dice_tc'):.3f} | {g('dice_et'):.3f} | **{s['dice_mean']:.3f}** "
            f"| {g('hd95_wt'):.1f} | {g('hd95_tc'):.1f} | {g('hd95_et'):.1f} | {g('sensitivity_et'):.3f} | {g('precision_et'):.3f} "
            f"| {e['params_m']} | {e['gflops_per_window']} | {e['predict_s_per_case']} | {e['peak_gpu_gb']} | {e['train_s_per_step']} |"
        )
    note = ("\nDice: overlap with the expert outline (1 = perfect). HD95: distance between predicted and true tumour edges, mm "
            "(lower is better; a missed or invented region counts as 373 mm). Means over cases.\n")
    Path("results/comparison.md").write_text("\n".join(lines) + "\n" + note)
    Path("results/comparison.json").write_text(json.dumps({n: results[n][0] for n in names}, indent=2))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / len(names)
    for j, n in enumerate(names):
        data = [[r["dice"] for r in results[n][1] if r["region"] == region] for region in REGIONS]
        pos = [i + (j - (len(names) - 1) / 2) * width for i in range(len(REGIONS))]
        c = COLORS[n]
        bp = ax.boxplot(data, positions=pos, widths=width * 0.85, patch_artist=True, showfliers=True,
                        boxprops={"facecolor": c, "edgecolor": c}, medianprops={"color": "white", "linewidth": 2},
                        whiskerprops={"color": c, "linewidth": 1.5}, capprops={"color": c, "linewidth": 1.5},
                        flierprops={"marker": "o", "markerfacecolor": c, "markeredgecolor": "white", "markersize": 6})
        ax.plot([], [], "s", color=c, label=NICE[n])
    ax.set_xticks(range(len(REGIONS)), ["Whole tumour (WT)", "Tumour core (TC)", "Active tumour (ET)"])
    ax.set_ylabel("Dice per brain")
    ax.set_ylim(0, 1.02)
    n_cases = len(results[names[0]][1]) // len(REGIONS)
    ax.set_title(f"Test set ({n_cases} brains): Dice per brain")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig("results/test_dice_boxplot.png", dpi=150)
    print("\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--limit", type=int, help="only the first N cases (quick check)")
    ap.add_argument("--overlap", type=float, default=0.5, help="sliding-window overlap (same as validation)")
    ap.add_argument("--save-preds", action="store_true", help="save predicted label maps (NIfTI) to output/preds/")
    ap.add_argument("--plots-only", action="store_true",
                    help="redraw the comparison table and boxplot from the saved test results (no prediction)")
    args = ap.parse_args()
    if args.plots_only:
        results = {}
        for m in args.models:
            summary = json.loads(Path(f"results/{m}/test_summary.json").read_text())
            rows = [{**r, "dice": float(r["dice"])} for r in csv.DictReader(open(f"results/{m}/test_per_case.csv"))]
            results[m] = (summary, rows)
        compare(results)
        return

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("brain-npr")
    cases = get_cases(args.split, args.limit)
    results = {m: evaluate_model(m, cases, args, torch.device("cuda")) for m in args.models}
    if len(results) > 1 and args.split == "test" and not args.limit:
        compare(results)


if __name__ == "__main__":
    main()
