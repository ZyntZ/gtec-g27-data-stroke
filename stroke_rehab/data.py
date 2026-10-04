"""Strict input validation and trial reconstruction for organizer MAT recordings."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

SAMPLE_RATE = 256
N_CHANNELS = 16
TRIAL_SECONDS = 8
EXPECTED_TRIALS = 80


@dataclass(frozen=True)
class Recording:
    signal: np.ndarray  # (samples, channels); units are left as recorded
    labels: np.ndarray  # one label per cue block; +1 left, -1 right
    onsets: np.ndarray  # first sample of the eight-second trigger block
    fs: int


def read_recording(path: str | Path, *, strict: bool = True) -> Recording:
    """Read one file. Labels come from block starts, NOT every nonzero sample.

    The organizer stores each trial's label on 2048 consecutive samples. The
    instruction appears two seconds after block onset. Do not train on pre-cue
    samples or on labels from the test run.
    """
    path = Path(path)
    values = loadmat(path, variable_names=("fs", "trig", "y"))
    if not all(k in values for k in ("fs", "trig", "y")):
        raise ValueError(f"Missing fs, trig or y in {path}")
    fs_value = float(np.asarray(values["fs"]).squeeze())
    if not fs_value.is_integer() or fs_value <= 0:
        raise ValueError(f"Invalid sampling rate: {fs_value}")
    fs = int(fs_value)
    signal = np.asarray(values["y"], dtype=np.float64)
    trig = np.asarray(values["trig"]).reshape(-1)
    if signal.ndim != 2 or signal.shape[0] != len(trig):
        raise ValueError("Signal and trigger lengths/shapes do not agree")
    if signal.shape[1] != N_CHANNELS or not np.isfinite(signal).all():
        raise ValueError("Expected 16 finite EEG channels")
    if not np.isin(trig, (-1, 0, 1)).all():
        raise ValueError("Unexpected trigger values")
    active = trig != 0
    onsets = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    offsets = np.flatnonzero(active & ~np.r_[active[1:], False]) + 1
    lengths = offsets - onsets
    labels = trig[onsets].astype(np.int8)
    if not np.all([np.all(trig[a:b] == label)
                   for a, b, label in zip(onsets, offsets, labels)]):
        raise ValueError("Trigger label changed within a trial")
    if len(onsets) == 0 or np.any(onsets[1:] < offsets[:-1]):
        raise ValueError("Empty or overlapping trigger blocks")
    if strict and (fs != SAMPLE_RATE or len(onsets) != EXPECTED_TRIALS
                   or np.any(lengths != TRIAL_SECONDS * fs)
                   or not np.array_equal(np.unique(labels), [-1, 1])):
        raise ValueError("Unexpected sampling rate, trial count, length or labels")
    return Recording(signal, labels, onsets, fs)


def session_paths(root: str | Path):
    root = Path(root)
    for patient in ("P1", "P2", "P3"):
        for stage in ("pre", "post"):
            train = root / f"{patient}_{stage}_training.mat"
            test = root / f"{patient}_{stage}_test.mat"
            if not train.is_file() or not test.is_file():
                raise FileNotFoundError(f"Missing {train} or {test}")
            yield patient, stage, train, test
