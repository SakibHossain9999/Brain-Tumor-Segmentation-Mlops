"""Make a fixed, reproducible train/val/test split of the MSD brain tumour cases."""
import json
import random
from pathlib import Path

ROOT = Path("data/msd/Task01_BrainTumour")
OUT = Path("splits/msd_split.json")
SEED = 42
N_VAL, N_TEST = 72, 72

meta = json.loads((ROOT / "dataset.json").read_text())
cases = sorted(Path(c["image"]).name.replace(".nii.gz", "") for c in meta["training"])

rng = random.Random(SEED)  # private random generator, so nothing else can change the order
rng.shuffle(cases)

split = {
    "seed": SEED,
    "test": sorted(cases[:N_TEST]),
    "val": sorted(cases[N_TEST:N_TEST + N_VAL]),
    "train": sorted(cases[N_TEST + N_VAL:]),
}

# safety checks: no case in two sets, nothing lost
assert not set(split["train"]) & set(split["val"])
assert not set(split["train"]) & set(split["test"])
assert not set(split["val"]) & set(split["test"])
assert len(split["train"]) + len(split["val"]) + len(split["test"]) == len(cases)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(split, indent=2))
print({k: len(v) for k, v in split.items() if k != "seed"}, "->", OUT)