"""Temperature scaling, triage and calibration metrics.

Shared by ModelService, scripts/calibrate.py and tests/. No Streamlit imports,
so it can be used outside the app.
"""
import json
from pathlib import Path

import numpy as np
import torch

DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "outputs" / "calibration.json"

# Calibrated max probability below this -> refer to a clinician
REFER_THRESHOLD = 0.9
STATUS_CONFIDENT = "confident"
STATUS_REFER = "refer"
REFER_MESSAGE = "refer: uncertain, clinician review recommended"


def load_temperature(path=DEFAULT_CALIBRATION_PATH):
    """Temperature from calibration.json, or None if the file is missing."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        temperature = float(json.load(f)["temperature"])
    if temperature <= 0:
        raise ValueError(f"Invalid temperature {temperature} in {path}")
    return temperature


def apply_temperature(logits, temperature):
    """Calibrated probabilities softmax(logits / T) over the last dimension."""
    return torch.softmax(torch.as_tensor(logits) / temperature, dim=-1)


def triage(confidence, threshold=REFER_THRESHOLD):
    """'confident' if the calibrated max probability reaches the threshold, else 'refer'."""
    return STATUS_CONFIDENT if confidence >= threshold else STATUS_REFER


def expected_calibration_error(confidences, correct, n_bins=15):
    """ECE with equal-width (lo, hi] bins over [0, 1]. Returns (ece, per-bin stats)."""
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(confidences, edges[1:-1], right=True), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = idx == b
        bins.append({"lo": edges[b], "hi": edges[b + 1], "n": int(m.sum()),
                     "acc": float(correct[m].mean()) if m.any() else None,
                     "conf": float(confidences[m].mean()) if m.any() else None})
    ece = sum(b["n"] * abs(b["acc"] - b["conf"]) for b in bins if b["n"]) / len(confidences)
    return float(ece), bins
