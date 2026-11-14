"""Loading cases from disk: MSD (one 4D file per case) or BraTS (one file per scan)."""
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

from .labels import regions_from_brats, regions_from_msd

# MSD channel order, from dataset.json: FLAIR, T1w, T1gd, T2w
MSD_CHANNELS = ("t2f", "t1n", "t1c", "t2w")

# BraTS file names: 2023 names and older (2021) names
BRATS_ALIASES = {
    "t1n": ["t1n", "t1"],
    "t1c": ["t1c", "t1ce"],
    "t2w": ["t2w", "t2"],
    "t2f": ["t2f", "flair"],
    "seg": ["seg"],
}


@dataclass
class Case:
    case_id: str
    affine: np.ndarray
    spacing: tuple
    images: dict = field(default_factory=dict)  # "t2f"/"t1n"/"t1c"/"t2w" -> float32 (H, W, D)
    regions: np.ndarray | None = None  # ground truth as (3, H, W, D) bool: WT, TC, ET


def _load(path):
    img = nib.as_closest_canonical(nib.load(str(path)))  # standard orientation (RAS)
    return img, np.asarray(img.dataobj)


def load_msd_case(case_id: str, root: str | Path) -> Case:
    """MSD Task01: imagesTr/<id>.nii.gz holds all 4 scans; labelsTr/<id>.nii.gz the label (if any)."""
    root = Path(root)
    image_path = next((root / d / f"{case_id}.nii.gz" for d in ("imagesTr", "imagesTs")
                       if (root / d / f"{case_id}.nii.gz").exists()), None)
    if image_path is None:
        raise FileNotFoundError(f"{case_id} not found in {root}/imagesTr or imagesTs")
    img, data = _load(image_path)
    data = data.astype(np.float32)  # (H, W, D, 4)
    case = Case(
        case_id=case_id,
        affine=img.affine.copy(),
        spacing=tuple(float(z) for z in img.header.get_zooms()[:3]),
        images={m: data[..., i] for i, m in enumerate(MSD_CHANNELS)},
    )
    label_path = root / "labelsTr" / f"{case_id}.nii.gz"
    if label_path.exists():
        case.regions = regions_from_msd(_load(label_path)[1])
    return case


def _find(case_dir: Path, aliases):
    for alias in aliases:
        for ext in (".nii.gz", ".nii"):
            hits = sorted(case_dir.glob(f"*[-_]{alias}{ext}"))
            if hits:
                return hits[0]
    return None


def load_brats_case(case_dir: str | Path) -> Case:
    """BraTS: one folder per case with *_flair / *_t1 / *_t1ce / *_t2 (/ *_seg) files."""
    case_dir = Path(case_dir)
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Case folder not found: {case_dir}")
    case, shape = None, None
    for modality, aliases in BRATS_ALIASES.items():
        path = _find(case_dir, aliases)
        if path is None:
            continue
        img, data = _load(path)
        if case is None:
            case = Case(case_dir.name, img.affine.copy(), tuple(float(z) for z in img.header.get_zooms()[:3]))
            shape = data.shape
        elif data.shape != shape:
            raise ValueError(f"{path.name} has shape {data.shape}, which doesn't match the other scans")
        if modality == "seg":
            case.regions = regions_from_brats(data)
        else:
            case.images[modality] = data.astype(np.float32)
    if case is None or not case.images:
        raise FileNotFoundError(f"No MRI scans found in {case_dir}")
    return case
