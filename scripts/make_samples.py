"""Package a few test brains as downloadable samples, so visitors can try the upload without
downloading the whole 7 GB dataset.

    python scripts/make_samples.py      -> output/samples/  (uploaded to Hugging Face at deployment)

Each sample: <case>.nii.gz (all 4 scans, MSD format) + <case>_seg.nii.gz (the expert outline, optional
to upload). Only TEST brains: none of the models saw them, so results are a fair check.
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path("data/msd/Task01_BrainTumour")
OUT = Path("output/samples")
WEB = Path("web/public/data/try.json")  # what the viewer's "Try your own scans" panel shows
# where visitors download the samples from, relative to the viewer page (works at a site root and in a sub-folder)
SAMPLES_URL = "samples"
SPLIT = json.loads(Path("splits/msd_split.json").read_text())

# test brains NOT in the gallery, chosen for variety (from results/*/test_per_case.csv)
SAMPLES = {
    "BRATS_075": "Typical tumour (127 mL)",
    "BRATS_156": "Large tumour (223 mL)",
    "BRATS_095": "Mostly swelling (36 mL)",
    "BRATS_238": "Small tumour (16 mL)",
}

README = """# Sample brains for the brain-tumour viewer

Four brains from the **test set** of this project: none of the three models saw them during
training, so the scores you get are a fair check.

| File | What it is |
|---|---|
| `<case>.nii.gz` | All 4 MRI scans in one file (FLAIR, T1, T1 + contrast, T2): upload this |
| `<case>_seg.nii.gz` | The expert outline (optional): upload it too to see Dice scores and where the model disagrees |

{table}

## Source and licence

From the **Medical Segmentation Decathlon**, Task01_BrainTumour (http://medicaldecathlon.com/),
which is derived from the BraTS 2016/2017 challenge data. Licence: **CC-BY-SA 4.0**. The files
are unchanged (only the label files were renamed to `<case>_seg.nii.gz`).

Simpson, A. L., et al. (2019). A large annotated medical image dataset for the development and
evaluation of segmentation algorithms. arXiv:1902.09063.
Antonelli, M., et al. (2022). The Medical Segmentation Decathlon. Nature Communications 13, 4128.
"""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    entries, rows = [], []
    for case, title in SAMPLES.items():
        assert case in SPLIT["test"], f"{case} is not a test brain"
        scans, seg = OUT / f"{case}.nii.gz", OUT / f"{case}_seg.nii.gz"
        shutil.copyfile(ROOT / "imagesTr" / f"{case}.nii.gz", scans)
        shutil.copyfile(ROOT / "labelsTr" / f"{case}.nii.gz", seg)
        entries.append({
            "case_id": case, "title": title, "split": "test",
            "scans": {"file": scans.name, "mb": round(scans.stat().st_size / 1e6, 1), "sha256": sha256(scans)},
            "label": {"file": seg.name, "kb": round(seg.stat().st_size / 1e3), "sha256": sha256(seg)},
        })
        rows.append(f"| {case} | {title} | {entries[-1]['scans']['mb']} MB |")
    table = "| Case | Tumour | Scan file |\n|---|---|---|\n" + "\n".join(rows)
    (OUT / "README.md").write_text(README.format(table=table))
    (OUT / "samples.json").write_text(json.dumps({"licence": "CC-BY-SA 4.0",
                                                  "source": "Medical Segmentation Decathlon, Task01_BrainTumour",
                                                  "samples": entries}, indent=2))
    WEB.parent.mkdir(parents=True, exist_ok=True)
    WEB.write_text(json.dumps({
        "samples_url": SAMPLES_URL,
        "samples": [{"case_id": e["case_id"], "title": e["title"], "scans": e["scans"]["file"],
                     "scans_mb": e["scans"]["mb"], "label": e["label"]["file"]} for e in entries],
        "counts": {k: len(v) for k, v in SPLIT.items() if isinstance(v, list)},
        "test_cases": SPLIT["test"],
        "unlabelled": 266,
    }, indent=1))
    print(f"wrote {len(entries)} samples to {OUT} ({sum(e['scans']['mb'] for e in entries):.0f} MB of scans)")


if __name__ == "__main__":
    main()
