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
from scipy.signal import butter, sosfiltfilt
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
# Outcome of `stroke-rehab montage-check` on the organizer archive: only the
# two P1 POST recordings follow montage.png (see results/montage_check.csv).
MONTAGE_PNG_FILES = ("P1_post_training", "P1_post_test")


def layout_for(path):
    """Channel names, in file order, for one organizer recording."""
    stem = Path(path).stem
    return LAYOUTS["montage_png" if stem in MONTAGE_PNG_FILES else "paper"]


def _distances(layout):
    xy = np.array([GRID[name] for name in layout], dtype=float)
    return np.linalg.norm(xy[:, None] - xy[None], axis=-1)


def median_trial_correlation(recording, band=(8.0, 30.0)):
    """Median over trials of the channel correlation matrix (robust to artefacts)."""
    sos = butter(4, band, btype="bandpass", fs=recording.fs, output="sos")
    filtered = sosfiltfilt(sos, recording.signal, axis=0)
    span = slice(recording.fs, 8 * recording.fs)
    return np.median([np.corrcoef(filtered[onset:][span].T)
                      for onset in recording.onsets], axis=0)


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
            row["assumed_layout"] = ("montage_png" if Path(path).stem in MONTAGE_PNG_FILES
                                     else "paper")
            # Smallest 8-30 Hz amplitude marks the side nearest the reference
            # (right earlobe in the paper): an orientation cue, not a proof.
            sos = butter(4, (8.0, 30.0), btype="bandpass", fs=recording.fs, output="sos")
            filtered = sosfiltfilt(sos, recording.signal, axis=0)
            amplitude = np.median([filtered[o:o + 8 * recording.fs].std(0)
                                   for o in recording.onsets], axis=0)
            row["lowest_amplitude_channel"] = layout_for(path)[int(amplitude.argmin())]
            rows.append(row)
            print(f"{patient} {stage} {run}: best fit {row['best_layout']} "
                  f"(montage.png rho {row['montage_png_rho']:+.2f}, "
                  f"paper rho {row['paper_rho']:+.2f})", flush=True)
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
