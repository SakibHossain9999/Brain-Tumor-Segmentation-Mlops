"""Reading and checking uploads: good files pass, common mistakes get a clear message."""
import nibabel as nib
import numpy as np
import pytest

from api.uploads import UploadError, load_upload
from tests.conftest import fake_brain


def save_label(path, values_by_zone, brain, affine=np.eye(4)):
    """A label map: a few voxel blocks inside the brain with the given numbers."""
    lab = np.zeros(brain.shape, np.uint8)
    for i, v in enumerate(values_by_zone):
        lab[40 + 4 * i: 44 + 4 * i, 40:48, 36:44] = v
    nib.save(nib.Nifti1Image(lab, affine), path)
    return path


def test_msd_style_upload_is_read_in_the_training_scan_order(brain_files):
    case, info = load_upload(brain_files["msd"])
    assert info["format"] == "msd"
    assert list(case.images) == ["t2f", "t1n", "t1c", "t2w"]  # FLAIR, T1, T1 + contrast, T2
    assert np.array_equal(case.images["t1c"], brain_files["scans"][..., 2])


def test_brats_style_four_files_give_the_same_case(brain_files):
    a, _ = load_upload(brain_files["msd"])
    b, info = load_upload(brain_files["brats"])
    assert info["format"] == "brats"
    for name in a.images:
        assert np.array_equal(a.images[name], b.images[name])


def test_label_numbering_detection(brain_files):
    d = brain_files["dir"]
    # values 1/2/3: MSD (1 = swelling) when the scans came as one MSD file
    label = save_label(d / "lab.nii.gz", [1, 2, 3], brain_files["brain"])
    case, info = load_upload(brain_files["msd"], label)
    assert info["label_format"] == "msd"
    swelling_only = case.regions[0] & ~case.regions[1]
    assert swelling_only[40:44, 40:48, 36:44].all()  # block with value 1 is swelling in MSD
    assert case.regions[2][48:52, 40:48, 36:44].all()  # block with value 3 is active
    # a 4 can only be BraTS 2021 and earlier
    label4 = save_label(d / "lab4.nii.gz", [2, 1, 4], brain_files["brain"])
    _, info = load_upload(brain_files["msd"], label4)
    assert info["label_format"] == "brats"


@pytest.mark.parametrize("problem, message", [
    ("missing", "Missing scan"),
    ("3d_single", "must hold all 4 scans"),
    ("whole_head", "skull-stripped"),
    ("spacing", "1 mm"),
    ("not_nifti", "not a readable NIfTI"),
])
def test_common_mistakes_are_refused_with_a_reason(brain_files, problem, message):
    d = brain_files["dir"]
    scans = brain_files["scans"]
    if problem == "missing":
        args = ({k: v for k, v in brain_files["brats"].items() if k != "t2"},)
    elif problem == "3d_single":
        args = (brain_files["brats"]["flair"],)
    elif problem == "whole_head":
        noisy = scans.copy()
        noisy[noisy == 0] = 5.0  # background not removed
        nib.save(nib.Nifti1Image(noisy, np.eye(4)), d / "head.nii.gz")
        args = (d / "head.nii.gz",)
    elif problem == "spacing":
        nib.save(nib.Nifti1Image(scans, np.diag([2.0, 2.0, 2.0, 1.0])), d / "coarse.nii.gz")
        args = (d / "coarse.nii.gz",)
    else:
        (d / "notes.nii.gz").write_text("hello")
        args = (d / "notes.nii.gz",)
    with pytest.raises(UploadError, match=message):
        load_upload(*args)


def test_scans_of_different_sizes_are_refused(brain_files):
    files = dict(brain_files["brats"])
    other, _ = fake_brain(shape=(96, 96, 72))
    nib.save(nib.Nifti1Image(other[..., 1], np.eye(4)), brain_files["dir"] / "small_t1.nii.gz")
    files["t1"] = brain_files["dir"] / "small_t1.nii.gz"
    with pytest.raises(UploadError, match="size"):
        load_upload(files)
