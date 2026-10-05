"""Chunk-by-chunk Riemannian decoding: causal filters, running covariance, explicit state.

Run-level state is the filter state per band plus the classifier's reference
covariances. The active trial keeps running sums for one covariance per band,
not its EEG samples. A decision is emitted in the chunk that completes the
window and uses nothing after it. Offline, `riemann.filtered_trials(causal=True)`
followed by `window_covariances` gives the same covariances to rounding error.
"""
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from .riemann import BANDS, oas_from_covariance
from .spatial import CovarianceEvent
from .streaming import DecisionEvent


class CausalRiemannStream:
    """Shrunk covariance per band for one fixed window of each trial.

    Filters start at the first sample's steady state, as in
    `riemann.filtered_trials(causal=True)`. Onsets are reported only inside
    the chunk that contains them; at most one per chunk.
    """

    def __init__(self, fs=256, channels=16, window=(2.5, 3.5), bands=BANDS):
        start, stop = (round(t * fs) for t in window)
        if not 0 <= start < stop <= 8 * fs:
            raise ValueError("Window outside the eight-second trial")
        if fs <= 2 * max(high for _, high in bands):
            raise ValueError("Sampling rate below filter-bank Nyquist requirement")
        self.fs, self.channels = fs, channels
        self.start_offset, self.stop_offset = start, stop
        self.sos = [butter(4, band, btype="bandpass", fs=fs, output="sos") for band in bands]
        self.zi = None
        self.sample = 0
        self._trial = None
        self._last_onset = -1

    def process(self, chunk, *, onsets=()):
        """Return a CovarianceEvent when a trial's window completes, else None."""
        x = np.asarray(chunk, dtype=np.float64)
        if (x.ndim != 2 or x.shape[1] != self.channels or not len(x)
                or not np.isfinite(x).all()):
            raise ValueError("Expected a nonempty finite (samples, channels) EEG chunk")
        beginning, ending = self.sample, self.sample + len(x)
        onsets = tuple(onsets)
        if len(onsets) > 1:
            raise ValueError("At most one cue onset per chunk; use smaller chunks")
        if onsets:
            onset = onsets[0]
            if (not isinstance(onset, (int, np.integer)) or onset < beginning
                    or onset >= ending or onset <= self._last_onset or self._trial is not None):
                raise ValueError("Out-of-order, future, duplicate or overlapping trial onset")
            self._last_onset = int(onset)
            self._trial = dict(onset=int(onset), count=0,
                               mean=np.zeros((len(self.sos), self.channels)),
                               scatter=np.zeros((len(self.sos), self.channels, self.channels)))
        if self.zi is None:
            self.zi = [sosfilt_zi(sos)[:, :, None] * x[0] for sos in self.sos]
        filtered = []
        for band, sos in enumerate(self.sos):
            y, self.zi[band] = sosfilt(sos, x, axis=0, zi=self.zi[band])
            filtered.append(y)
        event = None
        if self._trial is not None:
            trial = self._trial
            left = max(beginning, trial["onset"] + self.start_offset)
            right = min(ending, trial["onset"] + self.stop_offset)
            if right > left:
                values = np.stack([band[left - beginning:right - beginning]
                                   for band in filtered])
                n = values.shape[1]
                chunk_mean = values.mean(axis=1)
                centered = values - chunk_mean[:, None, :]
                old_n, total = trial["count"], trial["count"] + n
                delta = chunk_mean - trial["mean"]
                trial["scatter"] += (np.einsum("bti,btj->bij", centered, centered)
                                     + np.einsum("bi,bj->bij", delta, delta) * (old_n * n / total))
                trial["mean"] += delta * (n / total)
                trial["count"] = total
            if ending >= trial["onset"] + self.stop_offset:
                if trial["count"] != self.stop_offset - self.start_offset:
                    raise ValueError("Incomplete trial: cue or EEG samples were skipped")
                covariance = np.array([oas_from_covariance(band / trial["count"], trial["count"])
                                       for band in trial["scatter"]])
                event = CovarianceEvent(trial["onset"], ending, covariance)
                self._trial = None
        self.sample = ending
        return event


class CausalRiemannDecoder:
    """Inference-only: one decision per trial from an already fitted model.

    Owns one run. Construction resets the model's adaptation state to the
    training reference; every decision then advances it. Do not share a model
    between two decoders that are running at the same time.
    """

    def __init__(self, fitted_model, *, fs=256, channels=16, window=(2.5, 3.5), bands=BANDS):
        if len(fitted_model.means_) != len(bands):
            raise ValueError("Model was fitted with a different number of bands")
        self.model = fitted_model.reset()
        self.stream = CausalRiemannStream(fs, channels, window, bands)

    def process(self, chunk, *, onsets=()):
        event = self.stream.process(chunk, onsets=onsets)
        if event is None:
            return None
        label = int(self.model.predict_next(event.covariance))
        return DecisionEvent(event.onset_sample, event.decision_sample, label)


def replay_decisions(recording, fitted_model, *, chunk_samples=64, window=(2.5, 3.5),
                     bands=BANDS):
    """Feed a recording in chunks, announcing each onset as it arrives; never scores."""
    if not isinstance(chunk_samples, int) or chunk_samples < 1:
        raise ValueError("chunk_samples must be a positive integer")
    decoder = CausalRiemannDecoder(fitted_model, fs=recording.fs,
                                   channels=recording.signal.shape[1],
                                   window=window, bands=bands)
    onsets = [int(o) for o in recording.onsets]
    pointer, decisions = 0, []
    for left in range(0, len(recording.signal), chunk_samples):
        right = min(left + chunk_samples, len(recording.signal))
        cues = []
        while pointer < len(onsets) and onsets[pointer] < right:
            cues.append(onsets[pointer])
            pointer += 1
        event = decoder.process(recording.signal[left:right], onsets=cues)
        if event is not None:
            decisions.append(event)
    if [d.onset_sample for d in decisions] != onsets:
        raise ValueError("Missing or out-of-order trial decision in the replay")
    return decisions


def run_stream_check(data_root, output_file, *, chunk_samples=64, window=(2.5, 3.5),
                     folds=5, purge=1):
    """Per session: does chunked streaming reproduce the offline causal decisions?

    The model is fitted on the training run. The test run is used WITHOUT its
    labels, only to compare three routes to the same decisions: all trials at
    once, trial by trial, and raw EEG in chunks. Training-only scores use the
    purged chronological folds of `diagnostics.temporal_folds`.
    """
    import csv
    from pathlib import Path

    from .data import read_recording, session_paths
    from .diagnostics import out_of_fold_score, temporal_folds
    from .features import BANDS as TEAM_BANDS
    from .riemann import RecenteredTangentSpace, filtered_trials, window_covariances

    def covariances(recording, bands=BANDS):
        return window_covariances(filtered_trials(recording, bands, causal=True),
                                  recording.fs, window)

    rows = []
    for patient, stage, train_path, test_path in session_paths(data_root):
        train, test = read_recording(train_path), read_recording(test_path)
        x_train = covariances(train)
        model = RecenteredTangentSpace().fit(x_train, train.labels)
        x_test = covariances(test)
        batch = model.predict(x_test)
        model.reset()
        one_by_one = np.array([model.predict_next(trial) for trial in x_test])
        decisions = replay_decisions(test, model, chunk_samples=chunk_samples, window=window)
        streamed = np.array([d.label for d in decisions])
        delays = [(d.decision_sample - d.onset_sample) / test.fs for d in decisions]
        blocks = temporal_folds(train.labels, n_splits=folds, purge=purge)
        row = dict(patient=patient, session=stage, test_trials=len(batch),
                   trial_by_trial_equals_batch=int((one_by_one == batch).sum()),
                   chunked_stream_equals_batch=int((streamed == batch).sum()),
                   chunk_samples=chunk_samples,
                   latest_decision_s=round(max(delays), 3))
        for name, x, adapt in (("train_blocked_cv_correct", x_train, True),
                               ("train_blocked_cv_correct_no_adaptation", x_train, False),
                               ("train_blocked_cv_correct_three_bands",
                                covariances(train, TEAM_BANDS), True)):
            row[name] = out_of_fold_score(x, train.labels,
                                          RecenteredTangentSpace(adapt=adapt), blocks)[0]
        row["train_trials"] = len(train.labels)
        rows.append(row)
        print(f"{patient} {stage}: stream = batch on {row['chunked_stream_equals_batch']}/"
              f"{row['test_trials']} unlabelled test trials; training-only blocked CV "
              f"{row['train_blocked_cv_correct']}/{row['train_trials']}", flush=True)
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
