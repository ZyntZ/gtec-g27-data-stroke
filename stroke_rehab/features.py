"""Post-cue filter-bank power and covariance features; no fitted test transforms."""
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

BANDS = ((8.0, 12.0), (12.0, 20.0), (20.0, 30.0))
# Cue is two seconds after trigger. All windows start at least 0.5 s later.
WINDOWS = ((2.5, 3.5), (3.5, 4.5), (4.5, 5.5), (5.5, 6.5))
CSP_WINDOW = (2.5, 6.5)


@dataclass(frozen=True)
class Features:
    power: np.ndarray  # trials x (bands * windows * channels)
    epochs: np.ndarray  # trials x bands x channels x time, for train-only CSP


def extract(recording) -> Features:
    fs = recording.fs
    if fs <= 2 * max(high for _, high in BANDS):
        raise ValueError("Sampling rate below filter-bank Nyquist requirement")
    bands = []
    power = []
    for low, high in BANDS:
        sos = butter(4, (low, high), btype="bandpass", fs=fs, output="sos")
        # Filtering the continuous recording avoids trial-edge artefacts.
        filtered = sosfiltfilt(sos, recording.signal, axis=0)
        band_epochs = np.stack([
            filtered[onset + round(CSP_WINDOW[0] * fs):
                     onset + round(CSP_WINDOW[1] * fs)].T
            for onset in recording.onsets
        ]).astype(np.float32)
        if band_epochs.shape[2] != round((CSP_WINDOW[1] - CSP_WINDOW[0]) * fs):
            raise ValueError("A trial extends beyond the end of a recording")
        bands.append(band_epochs)
        for start, stop in WINDOWS:
            lo = round((start - CSP_WINDOW[0]) * fs)
            hi = round((stop - CSP_WINDOW[0]) * fs)
            # A constant offset (including large DC offsets) cannot dominate.
            var = np.var(band_epochs[:, :, lo:hi], axis=-1, dtype=np.float64)
            power.append(np.log(np.maximum(var, 1e-12)))
    return Features(np.concatenate(power, axis=1), np.stack(bands, axis=1))
