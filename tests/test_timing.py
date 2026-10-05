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


def _feedback_recording(kind, fs=256, trials=40):
    """Synthetic feedback response at 3.5 s that the probes are NOT built to find."""
    rng = np.random.default_rng(9)
    onsets = fs * (2 + 10 * np.arange(trials))
    labels = np.tile([1, -1], trials // 2)
    signal = rng.normal(size=(onsets[-1] + 10 * fs, 4))
    time = np.arange(8 * fs) / fs
    for onset, label in zip(onsets, labels):
        start = 3.5 + (rng.uniform(0.0, 3.0) if kind == "jittered" else 0.0)
        sign = label if kind == "class_opposite" else 1
        burst = 4 * np.sin(2 * np.pi * 5 * (time - start)) * ((time >= start) & (time < start + 0.5))
        signal[onset:onset + 8 * fs] += sign * burst[:, None]
    return Recording(signal, labels, onsets, fs)


def test_negative_control_evoked_probe_nearly_misses_class_opposite_feedback():
    fixed = evoked_ratios(_feedback_recording("fixed"))["nominal_feedback_3p5s"]
    opposite = evoked_ratios(_feedback_recording("class_opposite"))["nominal_feedback_3p5s"]
    assert fixed > 8  # same response, same sign in both classes: found
    assert opposite < fixed / 4  # opposite sign per class: the pooled median nearly cancels it


def test_negative_control_evoked_probe_nearly_misses_jittered_onset():
    fixed = evoked_ratios(_feedback_recording("fixed"))["nominal_feedback_3p5s"]
    jittered = evoked_ratios(_feedback_recording("jittered"))["nominal_feedback_3p5s"]
    assert jittered < fixed / 4  # latency varies by up to 3 s: nothing survives the median


def test_per_hand_line_probe_sees_a_class_dependent_line_the_pooled_one_dilutes():
    recording = _recording()
    fs, time = recording.fs, np.arange(8 * 256) / 256
    signal = recording.signal.copy()
    for onset, label in zip(recording.onsets, recording.labels):
        if label == 1:  # stimulation harmonic only on left-hand trials
            signal[onset:onset + 8 * fs] += (3 * np.sin(2 * np.pi * 100 * time)
                                             * (time >= 3.5))[:, None]
    one_sided = Recording(signal, recording.labels, recording.onsets, fs)
    assert line_change_db(one_sided, 100.0, label=1) > 10
    assert abs(line_change_db(one_sided, 100.0, label=-1)) < 1.5
    assert line_change_db(one_sided, 100.0) < line_change_db(one_sided, 100.0, label=1)
