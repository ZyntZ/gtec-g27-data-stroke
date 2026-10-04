"""Training-run-only pre-instruction negative control and temporal label audit.

This is a diagnostic of how reliably EEG covariances track the trial labels;
it cannot identify the cause of any late-window decoding improvement.
"""
import csv
from pathlib import Path

import numpy as np
from scipy.signal import sosfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import read_recording
from .diagnostics import temporal_folds
from .features import BANDS
from .nested import _predict
from .spatial import CausalCovarianceStream, CovarianceCSP, replay_covariances


def pre_instruction_covariances(recording, window=(0.5, 1.75)):
    """Causal continuous filtering and per-trial pre-instruction covariance.

    Filtering reads each run forward in time. No sample after 1.75 s relative
    to any trial onset can affect that trial's returned covariance. The
    matched *classifier* is the same CSP+LDA used for the early post-cue run,
    but the negative-control interval is necessarily 1.25 rather than 1 s.
    """
    fs = recording.fs
    left, right = [round(float(t) * fs) for t in window]
    if not (0 <= left < right < 2 * fs):
        raise ValueError("Negative-control window must finish before the 2-second cue")
    onsets = np.asarray(recording.onsets)
    if (onsets.ndim != 1 or not len(onsets) or np.any(onsets < 0)
            or np.any(onsets + right > len(recording.signal))):
        raise ValueError("Missing or incomplete pre-instruction EEG window")
    matrices = []
    stream = CausalCovarianceStream(fs, recording.signal.shape[1])
    for sos in stream.sos:
        signal = sosfilt(sos, recording.signal, axis=0)
        epochs = np.stack([signal[onset + left:onset + right]
                           for onset in onsets])
        epochs -= epochs.mean(axis=1, keepdims=True)
        cov = np.einsum("nti,ntj->nij", epochs, epochs) / (right - left)
        matrices.append(cov)
    return np.stack(matrices, axis=1)


def blocked_permutation(labels, blocks, rng):
    """Shuffle labels within fixed chronological blocks, preserving counts.

    The chronological blocks are defined *before* shuffling, not from label
    identities; this preserves each block's base-rate and slow class drift.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, (-1, 1)).all():
        raise ValueError("Expected 1-D binary +/-1 trial labels")
    if not isinstance(blocks, int) or blocks < 2 or blocks > len(labels):
        raise ValueError("Expected at least two nonempty chronological blocks")
    shuffled = labels.copy()
    for block in np.array_split(np.arange(len(labels)), blocks):
        shuffled[block] = rng.permutation(labels[block])
    return shuffled


def fixed_csp_predictions(covariances, labels, folds):
    """Refit CSP, scaler and LDA on each fold's fit trials, never test labels."""
    labels = np.asarray(labels)
    covariances = np.asarray(covariances)
    if (covariances.ndim != 4 or len(covariances) != len(labels)
            or not np.isfinite(covariances).all()):
        raise ValueError("Unexpected or nonfinite covariance data")
    model = make_pipeline(CovarianceCSP(n_pairs=1, ridge=0.01),
                          StandardScaler(),
                          LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    output = np.empty(len(labels), dtype=np.int8)
    seen = np.zeros(len(labels), dtype=bool)
    for fit, validation in folds:
        if (np.intersect1d(fit, validation).size or seen[validation].any()
                or len(np.unique(labels[fit])) != 2):
            raise ValueError("Overlapping folds, duplicate validation or missing class")
        output[validation] = _predict(model, covariances, labels, fit, validation)
        seen[validation] = True
    if not seen.all():
        raise ValueError("Fold validation does not cover every trial")
    return output


def permutation_audit(covariances, labels, *, permutations=99, blocks=5,
                      purge=1, seed=27):
    """Fixed-model blocked CV and a one-sided, block-count-matched label null.

    Fold boundaries are identical in every shuffle. This *exploratory* null
    requires label exchangeability within blocks and does not make trials or
    patients independent. A +1 correction yields p >= 1/(permutations+1).
    """
    if not isinstance(permutations, int) or permutations < 1:
        raise ValueError("permutations must be a positive integer")
    labels = np.asarray(labels)
    folds = temporal_folds(labels, n_splits=blocks, purge=purge)
    observed = int(np.sum(fixed_csp_predictions(covariances, labels, folds) == labels))
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=int)
    for repeat in range(permutations):
        permuted = blocked_permutation(labels, blocks, rng)
        null[repeat] = int(np.sum(
            fixed_csp_predictions(covariances, permuted, folds) == permuted))
    p_upper = (1 + int(np.sum(null >= observed))) / (permutations + 1)
    p_lower = (1 + int(np.sum(null <= observed))) / (permutations + 1)
    p_two_sided = min(1.0, 2 * min(p_upper, p_lower))
    return observed, null, p_upper, p_lower, p_two_sided


def run_controls(data_dir, output_file, *, permutations=99, seed=27,
                 folds=5, purge=1, chunk_samples=64):
    """Compute fixed controls on the six *training* runs; do not read tests."""
    rows = []
    for patient in ("P1", "P2", "P3"):
        for session in ("pre", "post"):
            path = Path(data_dir) / f"{patient}_{session}_training.mat"
            recording = read_recording(path)
            before = pre_instruction_covariances(recording)
            after, _ = replay_covariances(
                recording, chunk_samples=chunk_samples, window=(2.5, 3.5))
            for phase, X in (("pre_instruction_0p5_1p75", before),
                             ("early_postcue_2p5_3p5", after)):
                score, null, p_upper, p_lower, p_two_sided = permutation_audit(
                    X, recording.labels, permutations=permutations,
                    blocks=folds, purge=purge, seed=seed)
                rows.append(dict(patient=patient, session=session, phase=phase,
                                 correct=score, trials=len(recording.labels),
                                 accuracy=score / len(recording.labels),
                                 null_median_correct=float(np.median(null)),
                                 null_p95_correct=float(np.percentile(null, 95)),
                                 p_upper=float(p_upper), p_lower=float(p_lower),
                                 p_two_sided=float(p_two_sided),
                                 permutations=permutations,
                                 seed=seed, folds=folds, purge_trials=purge))
                print(f"{patient} {session} {phase}: {score}/80, "
                      f"blocked-permutation upper-tail p={p_upper:.3f}, "
                      f"two-sided p={p_two_sided:.3f}", flush=True)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
