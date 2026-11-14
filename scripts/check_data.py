"""Check the data pipeline: shapes, label sanity, a picture, and whether loading keeps up with the GPU."""
import argparse
import sys
import time
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")  # save figures to file, no window needed (works over SSH)
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.data import REGIONS, get_loaders  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--limit", type=int, default=20, help="cases per split to cache (quick test)")
p.add_argument("--batches", type=int, default=60)
p.add_argument("--workers", type=int, default=8)
args = p.parse_args()

t0 = time.perf_counter()
train_loader, val_loader = get_loaders(limit=args.limit, num_workers=args.workers)
print(f"Cached {args.limit} train + {args.limit} val cases in {time.perf_counter() - t0:.1f} s")

batch = next(iter(train_loader))
x, y = batch["image"], batch["label"]
print("train image:", tuple(x.shape), x.dtype, "| label:", tuple(y.shape), y.dtype)
print("image mean/std inside crop:", round(x.mean().item(), 2), round(x.std().item(), 2))
for i, r in enumerate(REGIONS):
    print(f"  {r} voxels in this batch: {int(y[:, i].sum())}")
assert (y[:, 2] <= y[:, 1]).all() and (y[:, 1] <= y[:, 0]).all(), "regions are not nested!"
print("Nesting check passed: ET inside TC inside WT")

v = next(iter(val_loader))
print("val image:", tuple(v["image"].shape), "| label:", tuple(v["label"].shape))

# picture of one training crop: FLAIR, T1gd, and the three region masks (middle slice)
img, lab = x[0], y[0]
z = int(lab[0].sum(dim=(0, 1)).argmax()) if lab[0].any() else img.shape[-1] // 2
fig, ax = plt.subplots(1, 5, figsize=(20, 4))
ax[0].imshow(img[0, :, :, z].T, cmap="gray", origin="lower"); ax[0].set_title("FLAIR (crop)")
ax[1].imshow(img[2, :, :, z].T, cmap="gray", origin="lower"); ax[1].set_title("T1gd (crop)")
for i, r in enumerate(REGIONS):
    ax[2 + i].imshow(lab[i, :, :, z].T, cmap="gray", origin="lower"); ax[2 + i].set_title(r)
for a in ax:
    a.axis("off")
Path("output").mkdir(exist_ok=True)
fig.savefig("output/check_crop.png", dpi=80, bbox_inches="tight")
print("Saved output/check_crop.png")

# speed: how many batches per second can the loader deliver?
def forever(loader):
    while True:          # start the loader again whenever it runs out
        yield from loader

it = forever(train_loader)
next(it)
t0 = time.perf_counter()
for _ in range(args.batches):
    next(it)
s_per_batch = (time.perf_counter() - t0) / args.batches
print(f"\nData loading: {s_per_batch:.3f} s per batch with {args.workers} workers")
print("GPU needs (160^3, from the timing test): segformer3d 0.071, segresnet 0.321, swin_unetr 1.448 s/step")
print("Loading keeps up with a model if its s/batch is below that model's s/step.")
