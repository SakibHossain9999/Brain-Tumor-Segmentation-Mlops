"""Data loading and preprocessing for the MSD brain tumour dataset.

The fixed steps (load -> regions -> crop to brain -> normalise -> pad) are shared by
training, validation, testing and the API, so every stage sees identical inputs.
"""
import json
from pathlib import Path

import torch
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    CastToTyped,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    SpatialPadd,
)

DATA_ROOT = Path("data/msd/Task01_BrainTumour")
SPLIT_FILE = Path("splits/msd_split.json")
REGIONS = ("WT", "TC", "ET")
KEYS = ["image", "label"]


class ConvertMSDToRegionsd(MapTransform):
    """MSD labels -> 3 nested region masks (channels WT, TC, ET).

    MSD labels: 1 = edema, 2 = non-enhancing/necrotic, 3 = enhancing.
    Label 3 = enhancing was checked on the data: brightest on T1gd (+2.12 vs -0.09 and -0.25).
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            lab = d[key][0]  # drop the single channel: (H, W, D)
            wt = lab > 0
            tc = (lab == 2) | (lab == 3)
            et = lab == 3
            d[key] = torch.stack([wt, tc, et]).to(torch.uint8)  # (3, H, W, D)
        return d


def get_cases(split: str, limit: int | None = None, root=DATA_ROOT, split_file=SPLIT_FILE):
    """List of {"image", "label", "case_id"} dicts for one split ("train", "val" or "test")."""
    names = json.loads(Path(split_file).read_text())[split]
    if limit:
        names = names[:limit]
    return [
        {
            "image": str(Path(root) / "imagesTr" / f"{n}.nii.gz"),
            "label": str(Path(root) / "labelsTr" / f"{n}.nii.gz"),
            "case_id": n,
        }
        for n in names
    ]


def base_transforms(crop_size):
    """Fixed preprocessing: the same every time, so its result can be cached."""
    return [
        LoadImaged(keys=KEYS),
        EnsureChannelFirstd(keys=KEYS),                      # (H, W, D, 4) -> (4, H, W, D)
        ConvertMSDToRegionsd(keys="label"),                  # 0-3 labels -> WT, TC, ET masks
        CropForegroundd(keys=KEYS, source_key="image", allow_smaller=True),  # cut the black border
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),  # brain voxels -> mean 0, std 1
        SpatialPadd(keys=KEYS, spatial_size=crop_size),      # make every side at least crop_size
        CastToTyped(keys=KEYS, dtype=(torch.float32, torch.uint8)),
    ]


def train_transforms(crop_size):
    """Fixed steps, then random augmentation that is redone every time a case is used."""
    return Compose(
        base_transforms(crop_size)
        + [
            RandCropByPosNegLabeld(  # 50% crops centred on tumour, 50% anywhere
                keys=KEYS, label_key="label", spatial_size=crop_size, pos=1, neg=1, num_samples=1
            ),
            RandFlipd(keys=KEYS, prob=0.5, spatial_axis=0),
            RandFlipd(keys=KEYS, prob=0.5, spatial_axis=1),
            RandFlipd(keys=KEYS, prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),  # contrast +-10%
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),  # brightness +-0.1
        ]
    )


def eval_transforms(crop_size):
    """Validation / test: fixed steps only, nothing random."""
    return Compose(base_transforms(crop_size))


def get_loaders(crop_size=(160, 160, 160), batch_size=2, num_workers=8, cache_workers=16, limit=None):
    """Cached train and validation loaders. `limit` uses only the first N cases (for quick tests)."""
    train_ds = CacheDataset(get_cases("train", limit), train_transforms(crop_size), cache_rate=1.0, num_workers=cache_workers)
    val_ds = CacheDataset(get_cases("val", limit), eval_transforms(crop_size), cache_rate=1.0, num_workers=cache_workers)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)
    return train_loader, val_loader