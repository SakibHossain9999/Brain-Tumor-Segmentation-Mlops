"""Build the viewer's sample gallery: chosen test brains, each with the expert outline + all three models.

    PYTHONPATH=pipeline python scripts/build_gallery.py

Needs: the MSD data, output/preds/<model>/<case>.nii.gz (from ml/evaluate.py --save-preds),
results/<model>/test_per_case.csv and models/manifest.json (from ml/register.py).
Writes web/public/cases/<case>/ for each case and web/public/cases/index.json (the gallery list).
"""
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from brainviz.build import build_bundle
from brainviz.io import load_msd_case
from brainviz.labels import regions_from_msd

ROOT = Path("data/msd/Task01_BrainTumour")
OUT = Path("web/public/cases")
MODELS = ["segresnet", "swin_unetr", "segformer3d"]  # dropdown order: champion first

# Chosen from the test results for variety (largest, typical, mostly swelling, smallest, disagreement, failure).
# All are test brains: none of the models was trained or checked on them.
GALLERY = {
    "BRATS_155": ("Large tumour",
                  "The largest tumour in the test set (264 mL), with a big active rim. All three models outline it well."),
    "BRATS_212": ("Typical case",
                  "A medium-sized tumour (63 mL) where all three models score close to their average."),
    "BRATS_402": ("Mostly swelling",
                  "Most of the abnormal area (90 mL) is swelling around a small core. All three models do well."),
    "BRATS_376": ("Small tumour",
                  "Only 10 mL. SegResNet and Swin UNETR do well; SegFormer3D misses part of it."),
    "BRATS_176": ("A harsh score",
                  "This brain has no active tumour. SegResNet predicts a tiny 0.02 mL speck of it, so its active-tumour "
                  "Dice drops from 1 to 0, even though its overall outline is as good as the others'."),
    "BRATS_381": ("Where all models fail",
                  "The expert outlined 27 mL of tumour; all three models find only a small part of it, "
                  "and almost none of the core."),
}


def case_scores():
    """{model: {case: {region: {metric: value}}}} from the evaluation CSVs."""
    scores = {}
    for m in MODELS:
        per_case = {}
        for row in csv.DictReader(open(f"results/{m}/test_per_case.csv")):
            per_case.setdefault(row["case_id"], {})[row["region"]] = {
                k: (None if row[k] == "nan" else float(row[k])) for k in ("dice", "hd95", "sensitivity", "precision")
            }
        scores[m] = per_case
    return scores


def load_regions(path):
    img = nib.as_closest_canonical(nib.load(str(path)))  # same orientation as brainviz.io
    return regions_from_msd(np.asarray(img.dataobj))


def dice(a, b):
    total = int(a.sum() + b.sum())
    return 1.0 if total == 0 else 2 * int((a & b).sum()) / total


def main():
    manifest = json.loads(Path("models/manifest.json").read_text())
    scores = case_scores()
    index = []
    for case_id, (title, description) in GALLERY.items():
        case = load_msd_case(case_id, ROOT)
        segmentations = {"expert": {"regions": case.regions, "label": "Expert outline"}}
        for m in MODELS:
            regions = load_regions(f"output/preds/{m}/{case_id}.nii.gz")
            s = scores[m][case_id]
            # safety check: the prediction file on disk must give (almost) the same Dice as the evaluation.
            # Not exactly: the model predicts each region separately, so a few voxels can be "core" without
            # being "whole tumour". Saved as one label map, those count as whole tumour too, which shifts Dice
            # by ~0.001. A wrong or misaligned file would be off by far more than 0.01.
            for i, r in enumerate(("WT", "TC", "ET")):
                d = dice(regions[i], case.regions[i])
                if abs(d - s[r]["dice"]) > 0.01:
                    sys.exit(f"{case_id} {m} {r}: Dice from file {d:.4f} != evaluation {s[r]['dice']:.4f}")
            info = manifest["models"][m]
            segmentations[m] = {
                "regions": regions,
                "label": info["display_name"],
                "model_version": info["version"],
                "champion": m == manifest["champion"],
                "scores": s,  # this brain's Dice / HD95 / sensitivity / precision per region
                "dice_mean": round(float(np.mean([s[r]["dice"] for r in ("WT", "TC", "ET")])), 4),
            }
        build_bundle(case, OUT / case_id, segmentations, default="expert",
                     extra_meta={"title": title, "description": description, "split": "test"})
        index.append({
            "case_id": case_id, "title": title, "description": description,
            "tumour_ml": round(float(case.regions[0].sum()) * float(np.prod(case.spacing)) / 1000, 1),
            "dice_mean": {m: segmentations[m]["dice_mean"] for m in MODELS},
        })

    (OUT / "index.json").write_text(json.dumps({
        "champion": manifest["champion"],
        "models": {m: manifest["models"][m]["display_name"] for m in MODELS},
        "cases": index,
    }, indent=2))
    print(f"wrote {len(index)} cases + {OUT / 'index.json'}")


if __name__ == "__main__":
    main()
