import numpy as np
import pytest

from stroke_rehab.data import Recording
from stroke_rehab.features import extract


def test_feature_shape_and_offsets():
    rng = np.random.default_rng(27)
    fs = 256
    raw = rng.standard_normal((10 * fs, 16)) + 100_000.0
    rec = Recording(raw, np.array([1], dtype=np.int8), np.array([256]), fs)
    features = extract(rec)
    assert features.power.shape == (1, 3 * 4 * 16)
    assert features.epochs.shape == (1, 3, 16, 4 * fs)
    assert np.isfinite(features.power).all()


def test_missing_samples_fail():
    rec = Recording(np.zeros((1024, 16)), np.array([1]), np.array([500]), 256)
    with pytest.raises(ValueError, match="extends"):
        extract(rec)
