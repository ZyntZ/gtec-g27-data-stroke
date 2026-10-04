import numpy as np
import pytest
from scipy.io import savemat

from stroke_rehab.data import read_recording


def write_mat(path, *, invalid=False):
    trig = np.zeros(5200, dtype=np.int16)
    trig[200:2248] = 1
    trig[2700:4748] = -1
    if invalid:
        trig[250] = -1
    y = np.arange(5200 * 16, dtype=np.float64).reshape(5200, 16)
    savemat(path, {"fs": [[256.0]], "trig": trig[:, None], "y": y})


def test_trigger_blocks_not_samples(tmp_path):
    path = tmp_path / "one.mat"
    write_mat(path)
    record = read_recording(path, strict=False)
    assert record.onsets.tolist() == [200, 2700]
    assert record.labels.tolist() == [1, -1]
    assert record.signal.shape == (5200, 16)


def test_label_change_fails_closed(tmp_path):
    path = tmp_path / "corrupted.mat"
    write_mat(path, invalid=True)
    with pytest.raises(ValueError, match="changed"):
        read_recording(path, strict=False)


def test_missing_or_short_trials_rejected(tmp_path):
    path = tmp_path / "short.mat"
    write_mat(path)
    with pytest.raises(ValueError, match="Unexpected"):
        read_recording(path)
