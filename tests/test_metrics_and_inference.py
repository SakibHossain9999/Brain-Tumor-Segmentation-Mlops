"""Scores, dataset fingerprints, and the prediction wrapper (with a stand-in model)."""
import json

import numpy as np
import torch

from ml.fingerprint import identify, scan_hash
from ml.inference import predict, regions_to_labels
from ml.metrics import MISSED_HD95, region_metrics
from tests.conftest import fake_brain


def cube(shift=0, shape=(40, 40, 40)):
    m = np.zeros(shape, bool)
    m[10 + shift:20 + shift, 10:20, 10:20] = True
    return torch.from_numpy(m)


def test_perfect_prediction():
    s = region_metrics(cube(), cube(), [1, 1, 1])
    assert s["dice"] == 1 and s["hd95"] == 0 and s["sensitivity"] == 1 and s["precision"] == 1


def test_region_absent_and_predicted_absent_counts_as_perfect():
    empty = torch.zeros(40, 40, 40, dtype=torch.bool)
    assert region_metrics(empty, empty, [1, 1, 1])["dice"] == 1


def test_missed_region_scores_zero_and_the_worst_distance():
    empty = torch.zeros(40, 40, 40, dtype=torch.bool)
    s = region_metrics(empty, cube(), [1, 1, 1])
    assert s["dice"] == 0 and s["hd95"] == MISSED_HD95


def test_shifted_prediction():
    s = region_metrics(cube(shift=2), cube(), [1, 1, 1])
    assert abs(s["dice"] - 0.8) < 1e-6  # 8 of 10 slices overlap
    assert 1.5 <= s["hd95"] <= 2.5  # edges about 2 mm apart


def test_fingerprint_depends_on_values_not_on_how_they_are_stored(tmp_path):
    a = np.arange(1000, dtype=np.float32).reshape(10, 10, 10)
    assert scan_hash(a) == scan_hash(a.astype(np.int16))  # same values, other number type
    b = a.copy()
    b[5, 5, 5] += 1
    assert scan_hash(a) != scan_hash(b)
    table = tmp_path / "fp.json"
    table.write_text(json.dumps({"cases": {"BRATS_X": {"split": "test", "scans": {"t2f": scan_hash(a)}}}}))
    assert identify({"t2f": a}, str(table))["case_id"] == "BRATS_X"
    assert identify({"t2f": b}, str(table)) is None


def test_predict_crops_to_the_brain_and_returns_full_size():
    scans, brain = fake_brain(shape=(96, 96, 80))
    image = torch.from_numpy(np.moveaxis(scans, -1, 0))
    seen = []

    def stand_in_model(x):  # "tumour" everywhere it looks
        seen.append(tuple(x.shape))
        return torch.full((x.shape[0], 3, *x.shape[2:]), 5.0)

    cfg = {"data": {"crop_size": [64, 64, 64]}, "train": {"amp": False}}
    probs = predict(stand_in_model, image, cfg, sw_batch_size=1)
    assert probs.shape == (3, 96, 96, 80)
    assert all(s[2:] == (64, 64, 64) for s in seen)  # the model only ever sees full windows
    outside = ~torch.from_numpy(brain).any(-1) if False else ~torch.from_numpy(np.any(scans > 0, -1))
    box_has = probs[0] > 0.5
    # outside the brain's bounding box nothing is predicted
    idx = np.nonzero(np.any(scans > 0, -1))
    lo, hi = [i.min() for i in idx], [i.max() for i in idx]
    assert not box_has[: lo[0]].any() and not box_has[hi[0] + 1:].any()
    labels = regions_to_labels(probs > 0.5)
    assert labels.dtype == np.uint8 and labels.max() == 3
    assert outside.any()
