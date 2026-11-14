"""Label numbers -> regions: MSD and BraTS number the same zones differently."""
import numpy as np
import pytest

from brainviz.labels import regions_from_brats, regions_from_msd, zones_from_regions


def test_msd_numbering():
    # MSD: 1 swelling, 2 inactive core, 3 active
    wt, tc, et = regions_from_msd(np.array([0, 1, 2, 3]))
    assert wt.tolist() == [False, True, True, True]
    assert tc.tolist() == [False, False, True, True]
    assert et.tolist() == [False, False, False, True]


@pytest.mark.parametrize("active", [4, 3])  # BraTS 2021 and earlier use 4, BraTS 2023 uses 3
def test_brats_numbering_gives_the_same_regions(active):
    # BraTS: 1 inactive core, 2 swelling, 4 (or 3) active: the same tissue as MSD [0, 1, 2, 3] above
    brats = regions_from_brats(np.array([0, 2, 1, active]))
    msd = regions_from_msd(np.array([0, 1, 2, 3]))
    assert np.array_equal(brats, msd)


def test_unknown_label_values_are_refused():
    with pytest.raises(ValueError):
        regions_from_msd(np.array([0, 1, 5]))


def test_zones_are_nested_even_if_the_model_isnt():
    # a model can mark "active" where it did not mark "core": the zones must still be consistent
    wt = np.array([True, True, True, False])
    tc = np.array([False, True, False, False])
    et = np.array([False, True, True, False])  # last-but-one: active but not core
    zones = zones_from_regions(np.stack([wt, tc, et]))
    total = zones["edema"].astype(int) + zones["non_enhancing"] + zones["enhancing"]
    assert total.tolist() == wt.astype(int).tolist()  # every tumour voxel in exactly one zone
    assert not (zones["enhancing"] & ~tc).any()
