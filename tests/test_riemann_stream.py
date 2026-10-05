import numpy as np
import pytest
from sklearn.base import clone
from sklearn.covariance import oas

from stroke_rehab.data import Recording
from stroke_rehab.riemann import (RecenteredTangentSpace, filtered_trials, oas_from_covariance,
                                  window_covariances)
from stroke_rehab.riemann_stream import (CausalRiemannDecoder, CausalRiemannStream,
                                         StreamDecoderFailedError, replay_decisions)

WINDOW = (2.5, 3.5)


def _run(seed, gain=1.0, trials=24, fs=256, channels=4):
    rng = np.random.default_rng(seed)
    onsets = fs + 9 * fs * np.arange(trials) + rng.integers(0, fs, trials)
    labels = np.tile([1, -1], trials // 2)
    signal = rng.normal(size=(onsets[-1] + 8 * fs, channels)) + 300.0  # DC offset
    time = np.arange(8 * fs) / fs
    for onset, label in zip(onsets, labels):
        signal[onset:onset + 8 * fs, 0 if label == 1 else 1] += 2 * np.sin(2 * np.pi * 10 * time)
    signal[:, 2] *= gain
    return Recording(signal, labels, onsets, fs)


def _covariances(recording):
    return window_covariances(filtered_trials(recording, causal=True), recording.fs, WINDOW)


@pytest.fixture(scope="module")
def fitted():
    train = _run(1)
    return RecenteredTangentSpace(C=1.0).fit(_covariances(train), train.labels)


def test_oas_from_covariance_matches_scikit_learn():
    x = np.random.default_rng(0).normal(size=(256, 6)) @ np.random.default_rng(1).normal(size=(6, 6))
    assert np.allclose(oas_from_covariance(np.cov(x.T, bias=True), len(x)), oas(x)[0],
                       rtol=1e-10, atol=0)


def test_trial_by_trial_chunked_and_batch_decisions_agree(fitted):
    test = _covariances(_run(2, gain=3.0))
    batch, scores = fitted.predict(test), fitted.decision_function(test)
    fitted.reset()
    one_by_one = np.array([fitted.predict_next(trial) for trial in test])
    assert fitted.n_seen_ == len(test)
    fitted.reset()
    streamed_scores = np.array([fitted.decision_next(trial) for trial in test])
    fitted.reset()
    chunked = np.concatenate([fitted.predict_stream(test[:5]), fitted.predict_stream(test[5:6]),
                              fitted.predict_stream(test[6:])])
    assert np.array_equal(batch, one_by_one) and np.array_equal(batch, chunked)
    # Same features; the linear score differs only by floating-point summation order.
    assert np.allclose(scores, streamed_scores, rtol=1e-10, atol=1e-12)


def test_predict_is_stateless_and_reset_restores_the_training_reference(fitted):
    test = _covariances(_run(2, gain=3.0))
    fitted.reset()
    fitted.predict_stream(test[:7])
    seen, reference = fitted.n_seen_, [m.copy() for m in fitted.references_]
    first = fitted.predict(test)
    assert np.array_equal(first, fitted.predict(test))  # repeated calls agree
    assert fitted.n_seen_ == seen == 7
    assert all(np.array_equal(a, b) for a, b in zip(reference, fitted.references_))
    assert not np.allclose(fitted.references_[0], fitted.means_[0])  # the stream did adapt
    fitted.reset()
    assert fitted.n_seen_ == 0
    assert all(np.array_equal(a, b) for a, b in zip(fitted.references_, fitted.means_))


def test_without_reset_a_second_run_is_decoded_from_a_different_state(fitted):
    test = _covariances(_run(2, gain=3.0))
    fitted.reset()
    first = np.array([fitted.decision_next(trial) for trial in test])
    carried = np.array([fitted.decision_next(trial) for trial in test])  # no reset
    assert not np.allclose(first, carried)
    assert clone(fitted).get_params() == fitted.get_params()


@pytest.mark.parametrize("chunk_samples", [1, 64, 97, 1024])
def test_end_to_end_stream_matches_offline_causal_pipeline(fitted, chunk_samples):
    recording = _run(3, gain=3.0, trials=8)
    offline = _covariances(recording)
    stream = CausalRiemannStream(recording.fs, recording.signal.shape[1], WINDOW)
    events, pointer = [], 0
    for left in range(0, len(recording.signal), chunk_samples):
        right = min(left + chunk_samples, len(recording.signal))
        cues = [int(o) for o in recording.onsets if left <= o < right][:1]
        event = stream.process(recording.signal[left:right], onsets=cues)
        if event is not None:
            events.append(event)
    assert len(events) == len(recording.onsets)
    assert np.allclose(np.array([e.covariance for e in events]), offline, rtol=1e-8, atol=0)
    for event in events:  # decision in the chunk that completes the window, never earlier
        due = event.onset_sample + round(WINDOW[1] * recording.fs)
        assert due <= event.decision_sample < due + chunk_samples
    decisions = replay_decisions(recording, fitted, chunk_samples=chunk_samples, window=WINDOW)
    assert [d.label for d in decisions] == list(fitted.predict(offline))


def test_stream_decision_ignores_samples_after_the_window(fitted):
    recording = _run(3, gain=3.0, trials=8)
    changed = recording.signal.copy()
    last = recording.onsets[-1] + round(WINDOW[1] * recording.fs)
    changed[last:] += 1000.0  # alter everything after the last trial's window
    altered = Recording(changed, recording.labels, recording.onsets, recording.fs)
    a = replay_decisions(recording, fitted, window=WINDOW)
    b = replay_decisions(altered, fitted, window=WINDOW)
    assert [d.label for d in a] == [d.label for d in b]


def test_classifier_failure_is_terminal_and_requires_reconstruction(fitted, monkeypatch):
    decoder = CausalRiemannDecoder(fitted, channels=4, window=(0, 0.5))
    chunk = np.random.default_rng(12).normal(size=(64, 4))
    assert decoder.process(chunk, onsets=(0,)) is None
    references = [r.copy() for r in fitted.references_]

    def fail(_):
        raise RuntimeError("forced classifier exception")

    with monkeypatch.context() as patch:
        patch.setattr(fitted.model_, "predict", fail)
        with pytest.raises(StreamDecoderFailedError, match="terminal") as caught:
            decoder.process(chunk)
        assert str(caught.value.__cause__) == "forced classifier exception"
        assert decoder.failed and decoder.stream.sample == 128
        assert fitted.n_seen_ == 0
        for before, after in zip(references, fitted.references_, strict=True):
            np.testing.assert_array_equal(before, after)

    # Removing the injected failure does not make the consumed EEG retryable.
    with pytest.raises(StreamDecoderFailedError, match="known run boundary"):
        decoder.process(chunk)
    assert decoder.stream.sample == 128 and fitted.n_seen_ == 0

    recovered = CausalRiemannDecoder(fitted, channels=4, window=(0, 0.5))
    assert recovered.process(chunk, onsets=(0,)) is None
    event = recovered.process(chunk)
    assert not recovered.failed and event.onset_sample == 0
    assert event.decision_sample == 128 and fitted.n_seen_ == 1


def test_invalid_eeg_makes_decoder_terminal(fitted):
    decoder = CausalRiemannDecoder(fitted, channels=4)
    with pytest.raises(StreamDecoderFailedError, match="terminal") as caught:
        decoder.process(np.full((64, 4), np.nan))
    assert isinstance(caught.value.__cause__, ValueError)
    assert decoder.failed and fitted.n_seen_ == 0
    with pytest.raises(StreamDecoderFailedError, match="known run boundary"):
        decoder.process(np.zeros((64, 4)))
