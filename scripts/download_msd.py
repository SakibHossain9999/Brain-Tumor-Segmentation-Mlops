"""Download and extract the MSD Task01 BrainTumour dataset (~7 GB)."""
import argparse
from pathlib import Path

from monai.apps import DecathlonDataset

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path("data/msd"))
args = parser.parse_args()
args.root.mkdir(parents=True, exist_ok=True)

# download=True fetches the archive, checks its MD5 checksum and extracts it.
# cache_rate=0 stops MONAI from loading every scan into memory afterwards.
ds = DecathlonDataset(
    root_dir=str(args.root),
    task="Task01_BrainTumour",
    section="training",
    download=True,
    cache_rate=0.0,
    num_workers=4,
)
print(f"Done: {len(ds)} cases listed, files in {args.root / 'Task01_BrainTumour'}")
