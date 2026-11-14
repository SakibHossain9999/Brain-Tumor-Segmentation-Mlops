"""Shared test helpers. The tests use small synthetic volumes, so they need neither the dataset nor the models."""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "pipeline")]


def fake_brain(shape=(96, 96, 80), seed=0):
    """A skull-stripped 'brain': an ellipsoid of positive values on an exactly-zero background, 4 scans."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape], indexing="ij")
    brain = (zz / 0.8) ** 2 + (yy / 0.7) ** 2 + (xx / 0.6) ** 2 < 1
    scans = np.zeros((*shape, 4), np.float32)
    for c in range(4):
        scans[..., c][brain] = rng.uniform(100, 1000, brain.sum())
    return scans, brain


@pytest.fixture
def brain_files(tmp_path):
    """The fake brain saved MSD-style (one 4D file) and BraTS-style (four files)."""
    scans, brain = fake_brain()
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(scans, affine), tmp_path / "case.nii.gz")
    four = {}
    for i, slot in enumerate(["flair", "t1", "t1ce", "t2"]):
        four[slot] = tmp_path / f"case_{slot}.nii.gz"
        nib.save(nib.Nifti1Image(scans[..., i], affine), four[slot])
    return {"msd": tmp_path / "case.nii.gz", "brats": four, "scans": scans, "brain": brain, "dir": tmp_path}
