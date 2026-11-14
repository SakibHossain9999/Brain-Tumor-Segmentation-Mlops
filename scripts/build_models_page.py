"""Collect everything the viewer's Models page shows into one file: web/public/data/models.json.

    python scripts/build_models_page.py

Reads what training and evaluation already saved:
    results/<model>/history.csv          training curves
    results/<model>/test_summary.json    test scores + efficiency
    results/<model>/test_per_case.csv    per-brain scores (for the boxplot and the statistics)
    models/<model>.json, .onnx           ONNX export: CPU speed, agreement with PyTorch, file size
    models/manifest.json                 registry: versions, champion
"""
import csv
import itertools
import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

MODELS = ["segresnet", "swin_unetr", "segformer3d"]  # fixed order = fixed colour on the page
REGIONS = ["WT", "TC", "ET"]
# training loss is logged every 50 steps; average it over each 1,000 steps, so it has a point at the
# same steps as the validation loss (which was measured every 1,000 steps) and the two compare directly
TRAIN_EVERY = 1000
OUT = Path("web/public/data/models.json")


def num(x):
    return None if x in ("", "nan", None) else float(x)


def curves(name):
    rows = list(csv.DictReader(open(f"results/{name}/history.csv")))
    train = [(int(r["step"]), num(r["train_loss"])) for r in rows if num(r["train_loss"]) is not None]
    binned = {}
    for step, loss in train:
        binned.setdefault((step - 1) // TRAIN_EVERY, []).append(loss)
    train_curve = [{"step": (b + 1) * TRAIN_EVERY, "loss": round(float(np.mean(v)), 4)} for b, v in sorted(binned.items())]
    val = [
        {"step": int(r["step"]), "loss": round(num(r["val_loss"]), 4),
         **{k: round(num(r[f"val_dice_{k.lower()}"]), 4) for k in ("WT", "TC", "ET")},
         "mean": round(num(r["val_dice_mean"]), 4)}
        for r in rows if num(r["val_dice_mean"]) is not None
    ]
    return {"train": train_curve, "val": val}


def per_case(name):
    rows = list(csv.DictReader(open(f"results/{name}/test_per_case.csv")))
    cases = sorted({r["case_id"] for r in rows})
    table = {(r["case_id"], r["region"]): r for r in rows}
    return cases, {
        reg: [round(num(table[(c, reg)]["dice"]), 4) for c in cases] for reg in REGIONS
    }


def by_split(name):
    """Dice of the kept checkpoint on each set of brains, all measured by ml/evaluate.py the same way."""
    out = {}
    for split, suffix in (("train", "_train"), ("val", "_val"), ("test", "")):
        f = Path(f"results/{name}/test_summary{suffix}.json")
        if f.exists():
            s = json.loads(f.read_text())
            out[split] = {"n": s["n_cases"], **{r: s[f"dice_{r.lower()}"]["mean"] for r in REGIONS}, "mean": s["dice_mean"]}
    return out


def main():
    manifest = json.loads(Path("models/manifest.json").read_text())
    split = json.loads(Path("splits/msd_split.json").read_text())
    gallery = {c["case_id"] for c in json.loads(Path("web/public/cases/index.json").read_text())["cases"]}
    out_models, mean_dice = {}, {}
    case_ids = None
    for name in MODELS:
        summary = json.loads(Path(f"results/{name}/test_summary.json").read_text())
        onnx = json.loads(Path(f"models/{name}.json").read_text())
        eff = summary["efficiency"]
        case_ids, dice = per_case(name)
        mean_dice[name] = np.mean([dice[r] for r in REGIONS], axis=0)
        per_case_mean = {r: summary[f"dice_{r.lower()}"]["mean"] for r in REGIONS}
        out_models[name] = {
            "label": manifest["models"][name]["display_name"],
            "version": manifest["models"][name]["version"],
            "champion": name == manifest["champion"],
            "checkpoint_step": summary["checkpoint_step"],
            "val_dice_mean": summary["val_dice_mean"],
            "test": {
                "dice": {**per_case_mean, "mean": summary["dice_mean"]},
                "hd95_mean": {r: summary[f"hd95_{r.lower()}"]["mean"] for r in REGIONS},
                "hd95_median": {r: summary[f"hd95_{r.lower()}"]["median"] for r in REGIONS},
                "sensitivity": {r: summary[f"sensitivity_{r.lower()}"]["mean"] for r in REGIONS},
                "precision": {r: summary[f"precision_{r.lower()}"]["mean"] for r in REGIONS},
            },
            "efficiency": {
                "params_m": eff["params_m"],
                "gflops_per_window": eff["gflops_per_window"],
                "gpu_s_per_case": eff["predict_s_per_case"],
                "gpu_peak_gb": eff["peak_gpu_gb"],
                "cpu_s_per_case_2": onnx.get("cpu_s_per_case_2_threads"),
                "cpu_s_per_case_8": onnx.get("cpu_s_per_case_8_threads"),
                "onnx_mb": round(Path(f"models/{name}.onnx").stat().st_size / 1e6, 1),
                "onnx_vs_pytorch_dice": onnx.get("onnx_vs_pytorch_dice"),
                "train_s_per_step": eff["train_s_per_step"],
                "train_hours": round(eff["train_s_per_step"] * 30000 / 3600, 1),
                "train_note": eff.get("train_note", ""),
            },
            "curves": curves(name),
            "dice_by_split": by_split(name),
            "per_case_dice": dice,
        }

    # are the differences real? Wilcoxon signed-rank test on each brain's mean Dice (paired: same 72 brains)
    stats = []
    for a, b in itertools.combinations(MODELS, 2):
        da, db = mean_dice[a], mean_dice[b]
        stats.append({
            "a": a, "b": b,
            "p": float(wilcoxon(da, db).pvalue),
            "a_better": int((da > db).sum()), "b_better": int((db > da).sum()),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated": date.today().isoformat(),
        "registered_model": manifest["registered_model"],
        "champion": manifest["champion"],
        "champion_reason": manifest.get("champion_reason", ""),
        "split": {k: len(v) for k, v in split.items() if isinstance(v, list)},
        "training": {"steps": 30000, "crop": "160 x 160 x 160", "batch": 2, "optimiser": "AdamW, lr 1e-4, warm-up + cosine"},
        "case_ids": case_ids,
        "gallery_cases": sorted(gallery),
        "models": out_models,
        "stats": stats,
    }, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} KB)")
    for s in stats:
        print(f"  {s['a']} vs {s['b']}: p = {s['p']:.4f} ({s['a_better']} vs {s['b_better']} brains better)")


if __name__ == "__main__":
    main()
