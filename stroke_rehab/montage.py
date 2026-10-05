"""Channel layouts, and a data-driven check of which layout each file follows.

The archive's `montage.png` and the bundled feasibility paper (`StrokeRehab.pdf`)
list two DIFFERENT 16-channel layouts. Neighbouring scalp electrodes share more
signal than distant ones, so the inter-channel correlations show which one a
recording follows. Decoding does not depend on channel names; anything that
names a hemisphere or compares channels across sessions does.
"""
from pathlib import Path
import csv

import numpy as np
from itertools import combinations

from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import spearmanr

from .data import read_recording, session_paths

# Left is negative x; one unit is one 10-10 electrode step.
GRID = {"FC5": (-3, 1), "FC3": (-2, 1), "FC1": (-1, 1), "FCz": (0, 1),
        "FC2": (1, 1), "FC4": (2, 1), "FC6": (3, 1),
        "C5": (-3, 0), "C3": (-2, 0), "C1": (-1, 0), "Cz": (0, 0),
        "C2": (1, 0), "C4": (2, 0), "C6": (3, 0),
        "CP5": (-3, -1), "CP3": (-2, -1), "CP1": (-1, -1), "CPz": (0, -1),
        "CP2": (1, -1), "CP4": (2, -1), "CP6": (3, -1), "Pz": (0, -2)}
LAYOUTS = {
    "montage_png": ("FC3", "FCz", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4",
                    "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "Pz"),
    "paper": ("FC5", "FC1", "FCz", "FC2", "FC6", "C5", "C3", "C1", "Cz",
              "C2", "C4", "C6", "CP5", "CP1", "CP2", "CP6"),
}
# In both P1 PRE files columns 13 and 14 fit far better exchanged (rho -0.93
# instead of -0.80, matching the other sessions); C3 and C4 are not affected.
LAYOUTS["paper_cp5_cp1_swapped"] = (*LAYOUTS["paper"][:12], "CP1", "CP5",
                                    *LAYOUTS["paper"][14:])
# Outcome of `stroke-rehab montage-check` on the organizer archive (see
# results/montage_check.csv). Inferred from the signals, not confirmed by g.tec.
FILE_LAYOUT = {"P1_post_training": "montage_png", "P1_post_test": "montage_png",
               "P1_pre_training": "paper_cp5_cp1_swapped",
               "P1_pre_test": "paper_cp5_cp1_swapped"}


def layout_for(path):
    """Channel names, in file order, for one organizer recording."""
    return LAYOUTS[FILE_LAYOUT.get(Path(path).stem, "paper")]


def _distances(layout):
    xy = np.array([GRID[name] for name in layout], dtype=float)
    return np.linalg.norm(xy[:, None] - xy[None], axis=-1)


def median_trial_correlation(recording, band=(8.0, 30.0), trials=slice(None)):
    """Median over trials of the channel correlation matrix (robust to artefacts)."""
    sos = butter(4, band, btype="bandpass", fs=recording.fs, output="sos")
    filtered = sosfiltfilt(sos, recording.signal, axis=0)
    span = slice(recording.fs, 8 * recording.fs)
    return np.median([np.corrcoef(filtered[onset:][span].T)
                      for onset in recording.onsets[trials]], axis=0)


def layout_fit(correlation, layout, *, permutations=2000, seed=27):
    """Rank correlation of electrode distance with signal correlation.

    A correct layout gives a strongly negative value. The p value is the share
    of random channel relabellings that fit at least as well. A left-right
    mirrored layout fits identically, so this cannot check orientation.
    """
    upper = np.triu_indices(len(layout), 1)
    distance = _distances(layout)
    rho = spearmanr(distance[upper], correlation[upper]).statistic
    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    for i in range(permutations):
        order = rng.permutation(len(layout))
        null[i] = spearmanr(distance[np.ix_(order, order)][upper],
                            correlation[upper]).statistic
    best = np.where(np.eye(len(layout), dtype=bool), -np.inf, correlation).argmax(1)
    adjacent = int(sum(distance[i, j] < 1.5 for i, j in enumerate(best)))
    return float(rho), float((np.sum(null <= rho) + 1) / (permutations + 1)), adjacent


def swaps_improving_fit(correlation, layout):
    """How many of the 120 two-channel swaps fit better than the layout as given.

    Zero means the stated channel ORDER is locally the best one, not merely the
    right set of electrodes. Swaps of mirror-image pairs cannot be detected.
    """
    upper = np.triu_indices(len(layout), 1)
    distance = _distances(layout)
    stated = spearmanr(distance[upper], correlation[upper]).statistic
    better = 0
    for i, j in combinations(range(len(layout)), 2):
        order = np.arange(len(layout))
        order[[i, j]] = j, i
        swapped = spearmanr(distance[np.ix_(order, order)][upper],
                            correlation[upper]).statistic
        better += swapped < stated - 1e-12
    return int(better)


def mirrored(layout):
    """Column order of the left-right mirror image of a layout."""
    position = {GRID[name]: i for i, name in enumerate(layout)}
    return [position[-GRID[name][0], GRID[name][1]] for name in layout]


def global_search(correlation, layout, *, restarts=20, seed=27):
    """Heuristic search for a better-fitting channel order, from random starts.

    Each restart swaps channel pairs for as long as a swap improves the fit.
    Returns whether the best order found is the stated one, its mirror image
    (which fits identically), or something else, with that order's fit. This
    samples a tiny part of the 16! orders: it is a strong consistency check,
    not an exhaustive search, and not independent confirmation of electrode
    labels. It cannot tell left from right.
    """
    upper = np.triu_indices(len(layout), 1)
    distance = _distances(layout)
    fit = lambda order: spearmanr(distance[np.ix_(order, order)][upper],
                                  correlation[upper]).statistic
    rng = np.random.default_rng(seed)
    best_order, best = None, np.inf
    for _ in range(restarts):
        order = rng.permutation(len(layout))
        current = fit(order)
        while True:
            options = []
            for i, j in combinations(range(len(layout)), 2):
                trial = order.copy()
                trial[[i, j]] = order[[j, i]]
                options.append((fit(trial), trial))
            value, candidate = min(options, key=lambda item: item[0])
            if value >= current - 1e-12:
                break
            order, current = candidate, value
        if current < best:
            best_order, best = order, current
    found = ("stated" if np.array_equal(best_order, np.arange(len(layout)))
             else "mirror" if np.array_equal(best_order, mirrored(layout)) else "other")
    return found, float(best)


def notch_hz(recording):
    """Mains notch already applied to the file: 50 or 60 Hz, whichever is emptier."""
    frequency, density = welch(recording.signal[recording.onsets[0]:], recording.fs,
                               nperseg=4 * recording.fs, axis=0)
    level = {hz: np.median(density[np.abs(frequency - hz) <= 0.5]) for hz in (50, 60)}
    return min(level, key=level.get)


def run_montage_check(data_root, output_file, *, permutations=2000, seed=27):
    rows = []
    for patient, stage, train_path, test_path in session_paths(data_root):
        for run, path in (("training", train_path), ("test", test_path)):
            recording = read_recording(path)
            correlation = median_trial_correlation(recording)
            row = dict(patient=patient, session=stage, run=run)
            for name, layout in LAYOUTS.items():
                rho, p, adjacent = layout_fit(correlation, layout,
                                              permutations=permutations, seed=seed)
                row |= {f"{name}_rho": round(rho, 3), f"{name}_perm_p": round(p, 4),
                        f"{name}_adjacent_best_partner": adjacent}
            row["best_layout"] = min(LAYOUTS, key=lambda k: row[f"{k}_rho"])
            row["swaps_improving_best_layout"] = swaps_improving_fit(
                correlation, LAYOUTS[row["best_layout"]])
            assigned = layout_for(path)
            found, best = global_search(correlation, assigned, seed=seed)
            row["global_search_finds"] = found
            row["global_search_rho"] = round(best, 3)
            halves = [layout_fit(median_trial_correlation(recording, trials=half),
                                 assigned, permutations=1)[0]
                      for half in (slice(0, None, 2), slice(1, None, 2))]
            row["odd_even_trial_rho"] = "/".join(f"{h:.3f}" for h in halves)
            row["notch_hz"] = notch_hz(recording)
            row["assumed_layout"] = FILE_LAYOUT.get(Path(path).stem, "paper")
            # Smallest 8-30 Hz amplitude marks the side nearest the reference
            # (right earlobe in the paper): an orientation cue, not a proof.
            sos = butter(4, (8.0, 30.0), btype="bandpass", fs=recording.fs, output="sos")
            filtered = sosfiltfilt(sos, recording.signal, axis=0)
            amplitude = np.median([filtered[o:o + 8 * recording.fs].std(0)
                                   for o in recording.onsets], axis=0)
            row["lowest_amplitude_channel"] = layout_for(path)[int(amplitude.argmin())]
            rows.append(row)
            print(f"{patient} {stage} {run}: best fit {row['best_layout']} ("
                  + ", ".join(f"{k} rho {row[f'{k}_rho']:+.2f}" for k in LAYOUTS)
                  + ")", flush=True)
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
