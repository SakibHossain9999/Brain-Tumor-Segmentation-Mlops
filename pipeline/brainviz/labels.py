"""Tumour regions and zones, and how to get them from each dataset's label numbers.

Inside the pipeline everything uses the same 3 regions the models predict (same as ml/data.py):
    WT = whole tumour,  TC = tumour core,  ET = enhancing (active) tumour
Each dataset's label numbers are converted to regions once, when loading.

The 3 zones drawn in the viewer are derived from the regions:
    edema          (swelling)        = WT and not TC
    non_enhancing  (inactive tumour) = TC and not ET
    enhancing      (active tumour)   = ET
"""
import numpy as np

REGION_NAMES = ("WT", "TC", "ET")

ZONES = {
    "edema": {"name": "Swelling (edema)", "color": [0.98, 0.78, 0.18]},
    "non_enhancing": {"name": "Inactive tumour (non-enhancing / necrotic)", "color": [0.55, 0.12, 0.65]},
    "enhancing": {"name": "Active tumour (enhancing)", "color": [0.90, 0.16, 0.20]},
}


def _as_labels(label, allowed):
    lab = np.rint(np.asarray(label)).astype(np.uint8)
    unknown = set(np.unique(lab).tolist()) - allowed
    if unknown:
        raise ValueError(f"Unexpected label values {sorted(unknown)}; expected {sorted(allowed)}")
    return lab


def regions_from_msd(label) -> np.ndarray:
    """MSD: 1 = edema, 2 = non-enhancing/necrotic, 3 = enhancing (checked on the data: 3 is brightest on T1gd)."""
    lab = _as_labels(label, {0, 1, 2, 3})
    return np.stack([lab > 0, np.isin(lab, (2, 3)), lab == 3])


def regions_from_brats(label) -> np.ndarray:
    """BraTS: 1 = necrotic, 2 = edema, enhancing = 4 (2021 and earlier) or 3 (2023+)."""
    lab = _as_labels(label, {0, 1, 2, 3, 4})
    return np.stack([lab > 0, np.isin(lab, (1, 3, 4)), np.isin(lab, (3, 4))])


def zones_from_regions(regions) -> dict:
    """(3, H, W, D) WT/TC/ET masks -> the 3 non-overlapping zones the viewer draws."""
    wt, tc, et = (np.array(r, dtype=bool) for r in regions)  # copies, so the input is not changed
    tc &= wt  # model outputs are not always perfectly nested: enforce ET inside TC inside WT
    et &= tc
    return {"edema": wt & ~tc, "non_enhancing": tc & ~et, "enhancing": et}
