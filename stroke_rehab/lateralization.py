"""Descriptive beta-power change over each motor cortex, per imagined hand.

Uses the per-file channel layout from `montage`. Only electrodes present in
BOTH layouts are used (C5 C3 C1 / C2 C4 C6), so PRE and POST are comparable.
Filtering is causal: the early window contains no EEG after its end. Three
patients, lesion side unknown: these numbers describe, they do not test, a
rehabilitation effect.
"""
from pathlib import Path
import csv

import numpy as np
from scipy.stats import mannwhitneyu

from .data import read_recording, session_paths
from .montage import layout_for
from .riemann import filtered_trials

BETA = ((13.0, 30.0),)
BASELINE = (0.5, 2.0)  # after the attention sound, before the instruction
WINDOWS = {"early_2p0_to_3p5s": (2.0, 3.5), "late_3p5_to_8p0s": (3.5, 8.0)}
SITES = {"C3": ("C5", "C1"), "C4": ("C2", "C6")}  # site minus mean of row neighbours


def power_change_db(path, window):
    """Per trial: beta power change (dB) at C3 and C4 relative to the baseline."""
    recording = read_recording(path)
    names = layout_for(path)
    index = names.index
    weights = np.zeros((len(SITES), len(names)))
    for row, (site, neighbours) in enumerate(SITES.items()):
        weights[row, index(site)] = 1.0
        weights[row, [index(n) for n in neighbours]] = -1.0 / len(neighbours)
    local = type(recording)(recording.signal @ weights.T, recording.labels,
                            recording.onsets, recording.fs)
    power = filtered_trials(local, BETA, causal=True)[:, 0] ** 2  # trial x time x site
    fs = recording.fs
    mean = lambda span: power[:, round(span[0] * fs):round(span[1] * fs)].mean(axis=1)
    return 10 * np.log10(mean(window) / mean(BASELINE)), recording.labels


def _median_ci(values, rng, draws=2000):
    resampled = np.median(rng.choice(values, (draws, len(values))), axis=1)
    return (float(np.median(values)), *np.percentile(resampled, (2.5, 97.5)).tolist())


def run_lateralization(data_root, output_file, *, seed=27):
    rng = np.random.default_rng(seed)
    rows, index_by_session = [], {}
    for patient, stage, train_path, test_path in session_paths(data_root):
        for interval, window in WINDOWS.items():
            parts = [power_change_db(p, window) for p in (train_path, test_path)]
            change = np.concatenate([p[0] for p in parts])
            labels = np.concatenate([p[1] for p in parts])
            # +1 is left-hand imagery: its opposite (contralateral) site is C4.
            contra_minus_ipsi = np.where(labels == 1, change[:, 1] - change[:, 0],
                                         change[:, 0] - change[:, 1])
            for hand, label in (("left", 1), ("right", -1)):
                pick = labels == label
                index_by_session[patient, stage, interval, hand] = contra_minus_ipsi[pick]
                row = dict(patient=patient, session=stage, interval=interval, hand=hand,
                           trials=int(pick.sum()))
                for name, values in (("c3_db", change[pick, 0]), ("c4_db", change[pick, 1]),
                                     ("contra_minus_ipsi_db", contra_minus_ipsi[pick])):
                    median, low, high = _median_ci(values, rng)
                    row |= {name: round(median, 2), f"{name}_ci_low": round(low, 2),
                            f"{name}_ci_high": round(high, 2)}
                rows.append(row)
    for row in rows:
        key = (row["patient"], row["interval"], row["hand"])
        pre = index_by_session[key[0], "pre", key[1], key[2]]
        post = index_by_session[key[0], "post", key[1], key[2]]
        row["pre_vs_post_mannwhitney_p"] = round(float(mannwhitneyu(pre, post).pvalue), 4)
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
