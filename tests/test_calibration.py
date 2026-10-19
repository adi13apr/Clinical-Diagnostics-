from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from services.calibration import (
    DEFAULT_CALIBRATION_PATH, REFER_THRESHOLD, STATUS_CONFIDENT, STATUS_REFER,
    apply_temperature, expected_calibration_error, load_temperature,
)

REPO = Path(__file__).resolve().parent.parent
LOGITS_CACHE = REPO / "data" / "cache" / "calibration_logits.npz"
LABELS = ["normal", "pneumonia", "tuberculosis"]  # notebook label order = ModelService.classes


def xray(mode="L"):
    """Small synthetic image; the stub model ignores the pixels."""
    return Image.new(mode, (64, 64), 128)


# (1) calibrated probabilities sum to 1 and argmax is unchanged vs raw

@pytest.mark.parametrize("temperature", [0.1318, 0.5, 1.0, 2.0])
def test_calibrated_probs_sum_to_one_and_keep_argmax(temperature):
    logits = torch.randn(500, 3, generator=torch.Generator().manual_seed(0), dtype=torch.float64) * 4
    probs = apply_temperature(logits, temperature)
    assert torch.allclose(probs.sum(dim=1), torch.ones(500, dtype=torch.float64))
    assert torch.equal(probs.argmax(dim=1), logits.argmax(dim=1))


def test_predict_returns_calibrated_probs(make_service, calibration_file):
    logits = [0.4, 1.1, -0.3]
    result = make_service(logits, calibration_file(0.25)).predict(xray())
    probs = [result["probabilities"][c] for c in ("Normal", "Pneumonia", "Tuberculosis")]
    assert sum(probs) == pytest.approx(1.0, abs=1e-6)
    assert result["diagnosis"] == "Pneumonia"  # raw argmax
    assert probs == pytest.approx(torch.softmax(torch.tensor(logits) / 0.25, 0).tolist(), abs=1e-6)
    assert result["confidence"] == pytest.approx(max(probs))
    assert result["temperature"] == 0.25


# (2) ECE on the cached test-split logits stays below 3%

def test_ece_on_cached_test_logits_below_3_percent():
    if not LOGITS_CACHE.exists():
        pytest.skip("No cached logits; run scripts/calibrate.py first")
    cache = np.load(LOGITS_CACHE, allow_pickle=False)
    if not {"labels", "splits"} <= set(cache.files):
        pytest.skip("Cached logits predate labels/splits; re-run scripts/calibrate.py")
    test = cache["splits"] == "test"
    labels = np.array([LABELS.index(l) for l in cache["labels"][test]])
    probs = apply_temperature(cache["logits"][test], load_temperature()).numpy()
    ece, _ = expected_calibration_error(probs.max(1), probs.argmax(1) == labels, n_bins=15)
    assert ece < 0.03


# (3) a low-confidence input returns "refer"

def test_low_confidence_input_is_referred(make_service, calibration_file):
    result = make_service([0.10, 0.05, 0.0], calibration_file(1.0)).predict(xray())
    assert result["confidence"] < REFER_THRESHOLD
    assert result["status"] == STATUS_REFER


def test_high_confidence_input_is_confident(make_service, calibration_file):
    result = make_service([2.0, 0.0, 0.0], calibration_file(0.1318)).predict(xray("RGB"))
    assert result["confidence"] >= REFER_THRESHOLD
    assert result["status"] == STATUS_CONFIDENT


def test_missing_calibration_falls_back_to_raw_and_refers(make_service, tmp_path):
    service = make_service([2.0, 0.0, 0.0], tmp_path / "missing.json")
    result = service.predict(xray())
    assert not service.calibrated and service.temperature == 1.0
    assert result["status"] == STATUS_REFER  # raw softmax(2, 0, 0) max = 0.79


def test_committed_calibration_file_loads():
    assert DEFAULT_CALIBRATION_PATH.exists()
    assert 0 < load_temperature() < 1  # the student model is underconfident, so T < 1


def test_ece_known_values():
    # Both samples in one bin: confidence 0.95 with accuracy 0.5 -> ECE 0.45; perfect calibration -> 0
    assert expected_calibration_error([0.95, 0.95], [1, 0])[0] == pytest.approx(0.45)
    assert expected_calibration_error([1.0, 1.0], [1, 1])[0] == pytest.approx(0.0)
