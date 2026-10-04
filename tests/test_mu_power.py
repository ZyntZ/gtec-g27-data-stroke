import importlib.util
from pathlib import Path

import numpy as np
import pytest


module_path = Path(__file__).resolve().parents[1] / "analysis/mu_power/mu_pass.py"
spec = importlib.util.spec_from_file_location("mu_pass", module_path)
mu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mu)


def test_known_signal_power_gain_and_offset():
    time = np.arange(256) / 256
    rhythm = np.sin(2 * np.pi * 10 * time)
    baseline = mu.mu_power(rhythm)
    assert mu.mu_power(rhythm + 1000) == pytest.approx(baseline)
    assert 10 * np.log10(mu.mu_power(2 * rhythm) / baseline) == pytest.approx(6.0205999133)


def test_test_recordings_are_rejected_before_io(monkeypatch):
    def forbidden_io(*args, **kwargs):
        pytest.fail("A test recording must never be opened by this analysis")

    monkeypatch.setattr(mu, "loadmat", forbidden_io)
    monkeypatch.setattr(mu, "sha256", forbidden_io)
    with pytest.raises(ValueError, match="restricted to the six named training files"):
        mu.read_training(Path("P1_pre_test.mat"))
