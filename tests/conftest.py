import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from services.model_handler import ModelService  # noqa: E402


class FixedLogitsModel(nn.Module):
    """Stands in for StudentEfficientNet: returns the same logits for every image."""

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, x):
        return self.logits.expand(x.shape[0], -1), None


@pytest.fixture
def calibration_file(tmp_path):
    """Writes a calibration.json with the given temperature and returns its path."""
    def write(temperature):
        path = tmp_path / "calibration.json"
        path.write_text(json.dumps({"temperature": temperature}), encoding="utf-8")
        return path
    return write


@pytest.fixture
def make_service(monkeypatch):
    """ModelService whose network is a FixedLogitsModel: no checkpoint, Streamlit cache or GPU needed."""
    def make(logits, calibration_path):
        monkeypatch.setattr(ModelService, "_load_model", lambda self, path: FixedLogitsModel(logits))
        return ModelService("unused.pth", calibration_path=calibration_path)
    return make
