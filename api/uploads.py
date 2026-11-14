"""Reading and checking uploaded scans.

Two ways to upload the 4 MRI scans:
    MSD style    one file holding all 4 scans (X x Y x Z x 4, in the order FLAIR, T1, T1 + contrast, T2)
    BraTS style  4 separate files, one per scan
plus, optionally, the expert outline (a label file), so the models can be scored against it.

The models expect BraTS-style preparation: skull removed (background exactly 0), the 4 scans aligned
to each other, about 1 mm voxels. Anything else is refused with a message saying what is wrong,
instead of silently producing a bad result.
"""
from pathlib import Path

import nibabel as nib
import numpy as np

from brainviz.io import MSD_CHANNELS, Case
from brainviz.labels import regions_from_brats, regions_from_msd

SCAN_SLOTS = {"flair": "t2f", "t1": "t1n", "t1ce": "t1c", "t2": "t2w"}  # upload field -> pipeline name
SCAN_NAMES = {"t2f": "FLAIR", "t1n": "T1", "t1c": "T1 + contrast", "t2w": "T2"}


class UploadError(ValueError):
    """A problem with the uploaded files, worded for the person who uploaded them."""


def _read(path: Path, what: str):
    try:
        img = nib.as_closest_canonical(nib.load(str(path)))  # standard orientation, as everywhere else
        data = np.asarray(img.dataobj)
    except Exception:
        raise UploadError(f"{what}: not a readable NIfTI file (.nii or .nii.gz).") from None
    if data.ndim == 4 and data.shape[-1] == 1:  # some tools save a 3D scan with an extra axis of size 1
        data = data[..., 0]
    return img, data


def _same_grid(img, ref_img, what: str, ref_what: str):
    if img.shape[:3] != ref_img.shape[:3]:
        raise UploadError(f"{what} has size {img.shape[:3]}, but {ref_what} has size {ref_img.shape[:3]}. "
                          "All scans must come from the same prepared case.")
    if not np.allclose(img.affine, ref_img.affine, atol=1e-2):
        raise UploadError(f"{what} and {ref_what} are positioned differently in space (their headers disagree). "
                          "The scans must be aligned to each other, as BraTS scans are.")


def check_scans(images: dict, spacing):
    """Refuse scans the models were not made for, with a reason."""
    shape = next(iter(images.values())).shape
    if any(s < 64 or s > 512 for s in shape):
        raise UploadError(f"The scans have size {shape}; that is not a brain MRI volume this tool can handle.")
    if not all(0.8 <= z <= 1.25 for z in spacing):
        raise UploadError(f"The voxels are {tuple(round(float(z), 2) for z in spacing)} mm. The models need about 1 mm "
                          "voxels (BraTS-style); resample the scans to 1 mm first.")
    masks = {}
    for name, vol in images.items():
        if not np.isfinite(vol).all():
            raise UploadError(f"The {SCAN_NAMES[name]} scan contains invalid values (NaN or infinity).")
        mask = vol != 0
        frac = mask.mean()
        if frac < 0.01:
            raise UploadError(f"The {SCAN_NAMES[name]} scan is (almost) empty.")
        # skull-stripped scans have an exactly-zero background around the brain; whole-head scans don't
        if frac > 0.6:
            raise UploadError(f"The {SCAN_NAMES[name]} scan doesn't look skull-stripped: {frac:.0%} of it is non-zero. "
                              "The models need brain-only scans with the background set to 0, as in BraTS.")
        masks[name] = mask
    # The four scans should cover the same brain. Measured on 30 test brains: scans of the same brain
    # overlap 70-100% (one scan can cover a smaller area), but scans of DIFFERENT brains also overlap up
    # to 100%, because BraTS fits every brain to the same template. So this only catches gross errors
    # (a flipped, shifted or unregistered scan); a scan from another patient cannot be detected this way.
    ref = masks["t2f"]
    for name, mask in masks.items():
        overlap = (mask & ref).sum() / max((mask | ref).sum(), 1)
        if overlap < 0.3:
            raise UploadError(f"The {SCAN_NAMES[name]} scan doesn't line up with the FLAIR scan (only {overlap:.0%} of "
                              "their brain areas overlap). Were the files mixed up, or not registered to each other?")


def load_upload(scans, label: Path | None = None, label_format: str = "auto", case_id: str = "upload"):
    """`scans`: one Path (MSD style) or {"flair": Path, "t1": Path, "t1ce": Path, "t2": Path} (BraTS style).
    Returns (Case, info). Raises UploadError with a readable message if something is wrong."""
    info = {}
    if isinstance(scans, (str, Path)):
        ref_img, data = _read(Path(scans), "The scan file")
        if data.ndim != 4 or data.shape[-1] != 4:
            raise UploadError(f"A single scan file must hold all 4 scans (size X x Y x Z x 4, in the order FLAIR, T1, "
                              f"T1 + contrast, T2), as in the MSD dataset. This file has size {data.shape}. "
                              "To upload 4 separate files, use the four scan slots instead.")
        images = {name: np.asarray(data[..., i], dtype=np.float32) for i, name in enumerate(MSD_CHANNELS)}
        info["format"] = "msd"
    else:
        missing = [SCAN_NAMES[SCAN_SLOTS[k]] for k in SCAN_SLOTS if not scans.get(k)]
        if missing:
            raise UploadError(f"Missing scan(s): {', '.join(missing)}. All four are needed: FLAIR, T1, T1 + contrast, T2.")
        images, ref_img = {}, None
        for slot, name in SCAN_SLOTS.items():
            img, data = _read(Path(scans[slot]), f"The {SCAN_NAMES[name]} file")
            if data.ndim != 3:
                raise UploadError(f"The {SCAN_NAMES[name]} file should hold one 3D scan, but has size {data.shape}.")
            if ref_img is None:
                ref_img = img
            else:
                _same_grid(img, ref_img, f"The {SCAN_NAMES[name]} scan", "the FLAIR scan")
            images[name] = data.astype(np.float32)
        info["format"] = "brats"

    spacing = tuple(float(z) for z in ref_img.header.get_zooms()[:3])
    check_scans(images, spacing)
    case = Case(case_id=case_id, affine=ref_img.affine.copy(), spacing=spacing, images=images)

    if label:
        lab_img, lab = _read(Path(label), "The expert outline file")
        if lab.ndim != 3:
            raise UploadError(f"The expert outline should be one 3D label map, but has size {lab.shape}.")
        _same_grid(lab_img, ref_img, "The expert outline", "the scans")
        values = set(np.unique(np.rint(lab)).astype(int).tolist())
        if not values <= {0, 1, 2, 3, 4}:
            raise UploadError(f"The expert outline has label values {sorted(values)}; expected 0-3 (MSD) or 0, 1, 2, 4 / "
                              "0-3 (BraTS).")
        # MSD and BraTS number the zones differently. 4 only exists in BraTS 2017-2021; otherwise use the
        # user's choice, or guess from how the scans were uploaded (one MSD file -> MSD labels).
        fmt = "brats" if 4 in values else (label_format if label_format in ("msd", "brats") else info["format"])
        case.regions = regions_from_msd(lab) if fmt == "msd" else regions_from_brats(lab)
        if not case.regions[0].any():
            raise UploadError("The expert outline is empty: it marks no tumour at all.")
        info["label_format"] = fmt
    return case, info
