"""Recognise brains from our own dataset when someone uploads them (data-leakage check).

A model scores unrealistically well on brains it was trained on, so the app tells the user
what an uploaded brain is:
    train / val   used during training -> scores look better than real-world performance
    test          held out: the models never saw it -> a fair check
    unlabelled    MSD's imagesTs: never used, but no expert outline exists
    (no match)    a brain from outside our dataset

How: a fingerprint (SHA-256 hash) of each scan's voxel values, after putting the scan in the
standard orientation. Re-compressing or renaming a file doesn't change it; any change to the
voxel values does (so a reprocessed version of the same patient, e.g. in BraTS 2021, is NOT
recognised - see the README).

    python ml/fingerprint.py                     # build splits/msd_fingerprints.json (~1 min)
    identify(scans)                              # in the API: scans = {"t2f": array, ...} or (4, H, W, D)
"""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path("data/msd/Task01_BrainTumour")
SPLIT_FILE = Path("splits/msd_split.json")
OUT = Path("splits/msd_fingerprints.json")
CHANNELS = ("t2f", "t1n", "t1c", "t2w")  # MSD order: FLAIR, T1, T1 + contrast, T2
MEANING = {
    "train": "used to train the models: scores will look better than real-world performance",
    "val": "used to pick the best checkpoint during training: scores will look somewhat better than real-world performance",
    "test": "held out: none of the models saw it during training, so this is a fair check",
    "unlabelled": "from the dataset's unlabelled part: never used by the models, but there is no expert outline to compare with",
}


def scan_hash(volume) -> str:
    """Fingerprint of one 3D scan: SHA-256 of its voxel values as float32 (orientation already standard)."""
    data = np.ascontiguousarray(np.asarray(volume, dtype=np.float32))
    return hashlib.sha256(data.tobytes()).hexdigest()


def _fingerprint_file(path: Path):
    img = nib.as_closest_canonical(nib.load(str(path)))  # same standard orientation as the pipeline
    data = np.asarray(img.dataobj, dtype=np.float32)
    return path.name.replace(".nii.gz", ""), [scan_hash(data[..., i]) for i in range(data.shape[-1])]


def build():
    split = json.loads(SPLIT_FILE.read_text())
    split_of = {case: name for name in ("train", "val", "test") for case in split[name]}
    # skip "._" files: hidden macOS leftovers inside the MSD archive, not scans
    files = [f for d in ("imagesTr", "imagesTs") for f in sorted((ROOT / d).glob("*.nii.gz"))
             if not f.name.startswith("._")]
    with ProcessPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(_fingerprint_file, files, chunksize=4))
    cases = {case: {"split": split_of.get(case, "unlabelled"), "scans": dict(zip(CHANNELS, hashes))}
             for case, hashes in results}
    OUT.write_text(json.dumps({
        "dataset": "Medical Segmentation Decathlon, Task01_BrainTumour",
        "split_file": str(SPLIT_FILE),
        "hash": "sha256 of float32 voxel values, per scan, after nibabel as_closest_canonical",
        "meaning": MEANING,
        "cases": cases,
    }, indent=1))
    counts = {s: sum(c["split"] == s for c in cases.values()) for s in MEANING}
    print(f"fingerprinted {len(cases)} brains {counts} -> {OUT}")


@lru_cache(maxsize=1)
def _lookup(path=str(OUT)):
    """scan hash -> (case id, split, scan name)."""
    table = {}
    for case, info in json.loads(Path(path).read_text())["cases"].items():
        for scan, h in info["scans"].items():
            table[h] = (case, info["split"], scan)
    return table


def identify(scans, fingerprints=str(OUT)):
    """Uploaded scans ({"t2f": array, ...} or a (4, H, W, D) array, standard orientation) ->
    {"case_id", "split", "meaning", "matched_scans"} if they come from our dataset, else None.
    One matching scan is enough: two different brains never have identical voxel values."""
    if not isinstance(scans, dict):
        scans = dict(zip(CHANNELS, scans))
    table = _lookup(fingerprints)
    hits = [table[h] for h in (scan_hash(v) for v in scans.values()) if h in table]
    if not hits:
        return None
    case_id, split, _ = hits[0]
    return {
        "case_id": case_id,
        "split": split,
        "meaning": MEANING[split],
        "matched_scans": sorted(s for c, _, s in hits if c == case_id),
    }


if __name__ == "__main__":
    build()
