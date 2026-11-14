"""Scoring one region of one brain against the expert outline. Shared by ml/evaluate.py and the API.

    region_metrics(pred, true, spacing) -> {"dice", "hd95", "sensitivity", "precision"}
"""
import torch
from monai.metrics import compute_hausdorff_distance

# BraTS convention: if a region is missing from the truth or the prediction (but not both), the edge
# distance is undefined; it is scored as the scan's diagonal (240 x 240 x 155 mm), the worst possible.
MISSED_HD95 = 373.13


def region_metrics(pred, true, spacing):
    """pred, true: boolean (H, W, D) masks (torch, on any device); spacing: voxel size in mm."""
    pred, true = torch.as_tensor(pred).bool(), torch.as_tensor(true).bool()
    tp = int((pred & true).sum())   # tumour, and predicted tumour
    fp = int((pred & ~true).sum())  # predicted tumour where there is none
    fn = int((~pred & true).sum())  # tumour the model missed
    n_pred, n_true = tp + fp, tp + fn
    if n_true == 0 and n_pred == 0:  # region absent and correctly predicted absent: perfect
        return {"dice": 1.0, "hd95": 0.0, "sensitivity": 1.0, "precision": 1.0}
    out = {
        "dice": 2 * tp / (n_pred + n_true),                       # overlap, 0..1
        "sensitivity": tp / n_true if n_true else float("nan"),   # share of the real tumour that was found
        "precision": tp / n_pred if n_pred else float("nan"),     # share of the prediction that is real tumour
    }
    if n_true == 0 or n_pred == 0:
        out["hd95"] = MISSED_HD95
    else:
        # only the box around both masks matters for edge distances: cropping makes this ~10x faster
        idx = torch.nonzero(pred | true)
        lo, hi = (idx.min(0).values - 1).clamp_min(0), idx.max(0).values + 2
        box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        p, t = pred[box][None, None].float(), true[box][None, None].float()
        # 95th percentile of the distances between the two edges, in mm (ignores the worst 5% stray voxels)
        out["hd95"] = float(compute_hausdorff_distance(p, t, include_background=True, percentile=95, spacing=spacing))
    return out
