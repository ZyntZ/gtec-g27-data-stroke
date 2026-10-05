"""Causal replay invariants; EEG data are synthetic and contain no patients."""

import numpy as np
import pytest
from scipy.signal import sosfilt

from stroke_rehab.data import Recording
from stroke_rehab.streaming import (CausalPowerStream, CausalTrialDecoder,
                                     PowerDecoderFailedError, replay_decisions, replay_features)


def fake_recording():
    fs = 256
    signal = np.random.default_rng(27).normal(size=(10 * fs, 16))
    return Recording(signal, np.array([1], dtype=np.int8), np.array([73]), fs)


def test_chunk_invariance_and_offline_causal_reference():
    rec = fake_recording()
    features, offsets, _ = replay_features(rec, chunk_samples=64)
    for chunk in (1, 31, 128, 511):
        other, latency, _ = replay_features(rec, chunk_samples=chunk)
        np.testing.assert_allclose(other, features, rtol=0, atol=1e-9)
        assert 3.5 <= latency[0] / rec.fs < 3.5 + chunk / rec.fs
    stream = CausalPowerStream()
    start = rec.onsets[0] + stream.start_offset
    stop = rec.onsets[0] + stream.stop_offset
    reference = np.concatenate([
        np.log(np.maximum(np.var(sosfilt(sos, rec.signal, axis=0)[start:stop], axis=0), 1e-12))
        for sos in stream.sos
    ])
    np.testing.assert_allclose(features[0], reference, rtol=0, atol=1e-9)
    assert offsets[0] >= stream.stop_offset


def test_future_eeg_cannot_change_first_decision_or_feature():
    rec = fake_recording()
    early, _, _ = replay_features(rec, chunk_samples=64)
    signal = rec.signal.copy()
    signal[rec.onsets[0] + round(3.5 * rec.fs):] *= 1e5
    changed, _, _ = replay_features(Recording(signal, rec.labels, rec.onsets, rec.fs),
                                    chunk_samples=64)
    np.testing.assert_array_equal(changed, early)


def test_reset_starts_each_recording_from_its_own_filter_state():
    rec = fake_recording()
    first, _, _ = replay_features(rec)
    second, _, _ = replay_features(rec)
    np.testing.assert_array_equal(first, second)


def test_no_early_event_and_reject_future_onset():
    stream = CausalPowerStream()
    rec = fake_recording()
    with pytest.raises(ValueError, match="onset"):
        stream.process(rec.signal[:64], onsets=[73])
    assert stream.sample == 0
    assert stream.process(rec.signal[:64]) is None
    # 73 is now in the next chunk: a late trigger is rejected.
    with pytest.raises(ValueError, match="onset"):
        stream.process(rec.signal[64:128], onsets=[63])
    assert stream.sample == 64
    assert stream.process(rec.signal[64:128], onsets=[73]) is None
    assert stream.process(rec.signal[128:256]) is None


def test_end_of_recording_and_overlapping_cues_fail_closed():
    rec = fake_recording()
    with pytest.raises(ValueError, match="Missing a trial"):
        replay_features(Recording(rec.signal[:500], rec.labels, rec.onsets, rec.fs))
    stream = CausalPowerStream()
    assert stream.process(rec.signal[:128], onsets=[73]) is None
    with pytest.raises(ValueError, match="onset"):
        stream.process(rec.signal[128:256], onsets=[160])


def test_fitted_decoder_emits_one_decision_after_window():
    from sklearn.dummy import DummyClassifier

    rec = fake_recording()
    X, reference_offsets, _ = replay_features(rec)
    estimator = DummyClassifier(strategy="constant", constant=1).fit(
        np.vstack([X[0], X[0] + 1]), np.array([-1, 1]))
    decoder = CausalTrialDecoder(estimator)
    assert decoder.process(rec.signal[:64]) is None
    decisions, times = replay_decisions(rec, estimator, chunk_samples=64)
    assert len(decisions) == 1 and decisions[0].label == 1
    assert decisions[0].decision_sample - decisions[0].onset_sample == reference_offsets[0]
    assert len(times) == (len(rec.signal) + 63) // 64
    assert np.all(times > 0)


def test_unfitted_decoder_is_rejected():
    from sklearn.dummy import DummyClassifier
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        CausalTrialDecoder(DummyClassifier())


def test_replay_features_never_use_cue_label_values():
    rec = fake_recording()
    changed = Recording(rec.signal, -rec.labels, rec.onsets, rec.fs)
    first, offsets, _ = replay_features(rec)
    other, other_offsets, _ = replay_features(changed)
    np.testing.assert_array_equal(first, other)
    np.testing.assert_array_equal(offsets, other_offsets)


def test_classifier_error_is_terminal_and_requires_fresh_replay(monkeypatch):
    from sklearn.dummy import DummyClassifier

    rec = fake_recording()
    X, _, _ = replay_features(rec)
    model = DummyClassifier(strategy="constant", constant=1).fit(
        np.vstack([X[0], X[0] + 1]), np.array([-1, 1]))
    decoder = CausalTrialDecoder(model)
    original_predict = model.predict

    def fail(_):
        raise RuntimeError("classifier failed")

    monkeypatch.setattr(model, "predict", fail)
    chunks = [rec.signal[i:i + 64] for i in range(0, len(rec.signal), 64)]
    for i, chunk in enumerate(chunks):
        if i == rec.onsets[0] // 64:
            cues = [int(rec.onsets[0])]
        else:
            cues = []
        if (i + 1) * 64 >= rec.onsets[0] + 3.5 * rec.fs:
            with pytest.raises(PowerDecoderFailedError, match="failed") as error:
                decoder.process(chunk, onsets=cues)
            assert isinstance(error.value.__cause__, RuntimeError)
            break
        assert decoder.process(chunk, onsets=cues) is None
    assert decoder.failed
    monkeypatch.setattr(model, "predict", original_predict)
    with pytest.raises(PowerDecoderFailedError, match="terminal"):
        decoder.process(chunks[i])  # A repaired model cannot make stale state safe.
    fresh, _ = replay_decisions(rec, model)
    assert len(fresh) == 1 and fresh[0].label == 1


@pytest.mark.parametrize("bad_output", [[0], [1, -1], [], [float("nan")]])
def test_invalid_classifier_output_never_emits_a_decision(monkeypatch, bad_output):
    from sklearn.dummy import DummyClassifier

    rec = fake_recording()
    X, _, _ = replay_features(rec)
    model = DummyClassifier(strategy="constant", constant=1).fit(
        np.vstack([X[0], X[0] + 1]), np.array([-1, 1]))
    monkeypatch.setattr(model, "predict", lambda _: bad_output)
    decoder = CausalTrialDecoder(model)
    with pytest.raises(PowerDecoderFailedError, match="failed"):
        for i in range(0, len(rec.signal), 64):
            decoder.process(rec.signal[i:i + 64],
                            onsets=[int(rec.onsets[0])] if i <= rec.onsets[0] < i + 64 else [])
    assert decoder.failed


def test_invalid_eeg_chunk_makes_decoder_terminal():
    from sklearn.dummy import DummyClassifier

    rec = fake_recording()
    X, _, _ = replay_features(rec)
    model = DummyClassifier(strategy="constant", constant=1).fit(
        np.vstack([X[0], X[0] + 1]), np.array([-1, 1]))
    decoder = CausalTrialDecoder(model)
    with pytest.raises(PowerDecoderFailedError, match="failed"):
        decoder.process(np.full((64, 16), np.nan))
    with pytest.raises(PowerDecoderFailedError, match="terminal"):
        decoder.process(rec.signal[:64])
