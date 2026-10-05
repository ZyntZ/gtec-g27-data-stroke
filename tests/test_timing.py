import numpy as np

from stroke_rehab.data import Recording
from stroke_rehab.timing import evoked_ratios, line_change_db


def _recording(add_line=False, add_evoked=False, fs=256, trials=20):
    rng = np.random.default_rng(4)
    onsets = fs * (2 + 10 * np.arange(trials))
    signal = rng.normal(size=(onsets[-1] + 10 * fs, 4))
    time = np.arange(8 * fs) / fs
    for onset in onsets:
        if add_line:  # 100 Hz stimulation harmonic switched on at 3.5 s
            signal[onset:onset + 8 * fs] += (3 * np.sin(2 * np.pi * 100 * time)
                                             * (time >= 3.5))[:, None]
        if add_evoked:  # a 5 Hz burst locked to 3.5 s
            signal[onset:onset + 8 * fs] += (4 * np.sin(2 * np.pi * 5 * (time - 3.5))
                                             * ((time >= 3.5) & (time < 4.0)))[:, None]
    return Recording(signal, np.tile([1, -1], trials // 2), onsets, fs)


def test_line_probe_detects_a_stimulation_harmonic_only_when_present():
    assert abs(line_change_db(_recording(), 100.0)) < 1.5
    assert line_change_db(_recording(add_line=True), 100.0) > 10


def test_evoked_probe_is_specific_to_the_event_that_has_a_response():
    quiet = evoked_ratios(_recording())
    assert all(0.4 < value < 2.5 for value in quiet.values())
    locked = evoked_ratios(_recording(add_evoked=True))
    assert locked["nominal_feedback_3p5s"] > 5 * locked["instruction_2p0s"]
