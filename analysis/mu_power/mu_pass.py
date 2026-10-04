"""A descriptive, training-only mu-power analysis of the organizer's stroke data.

No classifier is fitted. Test recordings are deliberately excluded. Windows are
chosen from task documentation, not from the appearance of these results.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import trapezoid
from scipy.io import loadmat
from scipy.signal import welch

CHANNELS = {"C3": 4, "C4": 8}  # Zero based; supplied montage.png: columns 5 and 9.
BAND_HZ = (8.0, 13.0)
WINDOWS = {"early": (2.0, 3.5), "full_task": (2.0, 8.0)}
BASELINE_S = (-1.0, 0.0)
PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT / "data" / "stroke-rehab"
DEFAULT_OUTPUT = PROJECT / "results" / "mu_power"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_training(path):
    """Read one training run and independently verify its trial/baseline structure."""
    path = Path(path).resolve()
    allowed = {f"P{p}_{stage}_training.mat" for p in (1, 2, 3)
               for stage in ("pre", "post")}
    if path.name not in allowed:
        raise ValueError("This pass is restricted to the six named training files.")
    digest = sha256(path)
    raw = loadmat(path, variable_names=["y", "fs", "trig"])
    if not {"y", "fs", "trig"} <= raw.keys():
        raise ValueError("Missing EEG, sample rate or trigger variable.")
    fs_values = np.asarray(raw["fs"])
    if fs_values.size != 1 or float(fs_values.item()) != 256.0:
        raise ValueError("This analysis expects the documented 256 Hz sample rate.")
    fs = 256
    eeg = np.asarray(raw["y"], dtype=float)
    trigger = np.asarray(raw["trig"]).reshape(-1)
    if eeg.ndim != 2 or eeg.shape != (len(trigger), 16):
        raise ValueError("Expected samples x 16 EEG channels and a matching trigger.")
    if not np.isfinite(eeg).all() or not np.isfinite(trigger).all():
        raise ValueError("Non-finite data need inspection, not silent removal.")
    if not set(np.unique(trigger)) <= {-1, 0, 1}:
        raise ValueError("Unexpected trigger code.")
    active = trigger != 0
    onsets = np.flatnonzero(active & np.r_[True, ~active[:-1]])
    offsets = np.flatnonzero(active & np.r_[~active[1:], True]) + 1
    if len(onsets) != 80 or len(offsets) != 80:
        raise ValueError("Expected exactly 80 held-label trial blocks.")
    if not np.all(offsets - onsets == 8 * fs):
        raise ValueError("Expected eight seconds per labelled block.")
    labels = trigger[onsets].astype(int)
    if np.sum(labels == 1) != 40 or np.sum(labels == -1) != 40:
        raise ValueError("Expected 40 left and 40 right trials.")
    for onset, offset, label in zip(onsets, offsets, labels):
        if not np.all(trigger[onset:offset] == label):
            raise ValueError("The label changes inside a block.")
        if onset < fs or not np.all(trigger[onset - fs:onset] == 0):
            raise ValueError("A one-second baseline overlaps a trial or file boundary.")
    return {"path": path, "sha256": digest, "fs": fs, "eeg": eeg,
            "trigger": trigger, "onsets": onsets, "labels": labels,
            "checks": {"finite_signal": True, "fs_256_hz": True,
                       "channels_16": True, "trials_80": True,
                       "left_40_right_40": True, "held_blocks_8_seconds": True,
                       "all_baselines_in_bounds_and_unlabelled": True}}


def mu_power(epochs, fs=256):
    """Welch density integrated over 8--13 Hz; last axis contains time samples.

    Segment-local detrending avoids DC offsets. No continuous zero-phase filter
    is used, so this estimator does not import samples from outside its window.
    The baseline has one 1-s segment; early has two; full-task has eleven.
    """
    epochs = np.asarray(epochs, dtype=float)
    if epochs.shape[-1] < fs:
        raise ValueError("At least one second is needed for this fixed estimator.")
    freq, density = welch(epochs, fs=fs, window="hann", nperseg=fs,
                          noverlap=fs // 2, nfft=fs, detrend="constant",
                          scaling="density", average="mean", axis=-1)
    selected = (freq >= BAND_HZ[0]) & (freq <= BAND_HZ[1])
    power = trapezoid(density[..., selected], x=freq[selected], axis=-1)
    if not np.isfinite(power).all() or np.any(power <= 0):
        raise ValueError("Mu power must be finite and positive; inspect invalid trials.")
    return power


def get_epochs(recording, start_s, stop_s):
    fs = recording["fs"]
    columns = list(CHANNELS.values())
    return np.stack([recording["eeg"][onset + int(start_s * fs):
                                        onset + int(stop_s * fs), columns].T
                     for onset in recording["onsets"]])


def analyze_training(path):
    recording = read_training(path)
    baseline = mu_power(get_epochs(recording, *BASELINE_S))
    windows = {}
    for name, interval in WINDOWS.items():
        task = mu_power(get_epochs(recording, *interval))
        windows[name] = {"power": task, "db": 10 * np.log10(task / baseline)}
    patient, stage, _ = recording["path"].stem.split("_")
    rows = []
    for trial, (onset, label) in enumerate(zip(recording["onsets"], recording["labels"]), 1):
        for column, channel in enumerate(CHANNELS):
            rows.append({"patient": patient, "stage": stage, "run": "training",
                         "trial": trial, "onset_sample_0based": int(onset),
                         "label": int(label), "hand": "left" if label == 1 else "right",
                         "channel": channel, "column_1based": CHANNELS[channel] + 1,
                         "baseline_mu_power_original_units_squared": float(baseline[trial - 1, column]),
                         "early_mu_power_original_units_squared": float(windows["early"]["power"][trial - 1, column]),
                         "full_task_mu_power_original_units_squared": float(windows["full_task"]["power"][trial - 1, column]),
                         "early_change_db": float(windows["early"]["db"][trial - 1, column]),
                         "full_task_change_db": float(windows["full_task"]["db"][trial - 1, column])})
    summaries = []
    for window in WINDOWS:
        for col, channel in enumerate(CHANNELS):
            for label, hand in [(1, "left"), (-1, "right")]:
                values = windows[window]["db"][recording["labels"] == label, col]
                q25, median, q75 = np.percentile(values, [25, 50, 75])
                summaries.append({"patient": patient, "stage": stage, "window": window,
                                  "channel": channel, "hand": hand, "n_trials": len(values),
                                  "median_db": float(median), "q25_db": float(q25),
                                  "q75_db": float(q75), "mean_db": float(np.mean(values)),
                                  "minimum_db": float(np.min(values)),
                                  "maximum_db": float(np.max(values))})
    # Recompute the first estimate by an independent hand-coded trapezoid sum.
    f, p = welch(get_epochs(recording, *BASELINE_S)[0, 0], fs=256,
                 window="hann", nperseg=256, noverlap=128, nfft=256,
                 detrend="constant", scaling="density", average="mean")
    selected = (f >= 8) & (f <= 13)
    x, y = f[selected], p[selected]
    independent = np.sum((y[1:] + y[:-1]) * np.diff(x) / 2)
    assert np.isclose(independent, baseline[0, 0], rtol=1e-12)
    assert sha256(recording["path"]) == recording["sha256"]
    return {"recording": recording, "baseline": baseline, "windows": windows,
            "rows": rows, "summaries": summaries,
            "checks": {**recording["checks"], "positive_finite_bandpower": True,
                       "independent_first_band_integral_agrees": True,
                       "source_hash_unchanged": True}}


def verify_estimator():
    t = np.arange(256) / 256
    ten = np.sin(2 * np.pi * 10 * t)
    p = mu_power(ten)
    ratio_db = float(10 * np.log10(mu_power(2 * ten) / p))
    checks = {"double_amplitude_is_6p0206_db": np.isclose(ratio_db, 6.020599913279624),
              "equal_power_is_zero_db": np.isclose(10 * np.log10(p / p), 0),
              "constant_offset_does_not_change_mu_power": np.isclose(mu_power(ten + 600), p),
              "25_hz_has_lower_mu_power_than_10_hz": mu_power(np.sin(2 * np.pi * 25 * t)) < p / 1000}
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise AssertionError(checks)
    return checks


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_p1(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {1: "#3274A1", -1: "#C76035"}
    paths = {}
    with plt.rc_context({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"}):
        for window, interval in WINDOWS.items():
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 6), sharey=True)
            fig.subplots_adjust(left=.11, right=.97, bottom=.27, top=.76, wspace=.15)
            fig.suptitle(f"P1 PRE training | mu power during {interval[0]:g}–{interval[1]:g} s",
                         x=.11, y=.96, ha="left", fontsize=17)
            fig.text(.11, .885, "8–13 Hz relative to each trial's −1–0 s baseline | one participant, one run",
                     fontsize=11)
            for column, (axis, channel) in enumerate(zip(axes, CHANNELS)):
                axis.axhline(0, color="#727272", linewidth=1, linestyle="--")
                for label, x in [(1, 0), (-1, 1)]:
                    values = result["windows"][window]["db"][result["recording"]["labels"] == label, column]
                    jitter = np.random.default_rng(27 + label + column).uniform(-.17, .17, len(values))
                    axis.scatter(x + jitter, values, s=27, color=colors[label], alpha=.75, edgecolors="none")
                    median = np.median(values)
                    axis.plot([x - .23, x + .23], [median, median], color="#202020", linewidth=2.5)
                    axis.text(x, 1.02, f"Median {median:+.2f} dB", ha="center", fontsize=10,
                              transform=axis.get_xaxis_transform())
                axis.set_title(f"{channel} · EEG column {CHANNELS[channel] + 1}", y=1.13)
                axis.set(xticks=[0, 1], xticklabels=["Left trials (+1)\nn = 40", "Right trials (−1)\nn = 40"],
                         xlim=(-.5, 1.5))
                axis.grid(axis="y", color="#ECECEC", linewidth=.6)
            axes[0].set_ylabel("Mu-power change (dB)\nnegative = less than baseline")
            fig.text(.11, .16, "Dots: individual trials. Black bars: medians. Dashed line: no change from baseline.", fontsize=10)
            caveat = ("Full-task window includes documented avatar/FES feedback; it does not isolate imagery."
                      if window == "full_task" else
                      "Early window ends at the diagram's approximate feedback boundary; actual onset is unverified.")
            fig.text(.11, .11, caveat, fontsize=10)
            fig.text(.11, .06, "All trials retained; no artifact rejection or continuous filtering. Descriptive, not a clinical effect.", fontsize=10, color="#555555")
            paths[window] = {}
            for extension in ("png", "svg"):
                target = Path(output) / f"p1_{window}_mu.{extension}"
                fig.savefig(target, dpi=180, facecolor="white")
                paths[window][extension] = str(target)
            plt.close(fig)
    return paths


def plot_session_extension(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.5), sharey=True)
    fig.subplots_adjust(left=.1, right=.97, bottom=.21, top=.83, hspace=.5, wspace=.2)
    colors = {"left": "#3274A1", "right": "#C76035"}
    all_values = np.concatenate([r["windows"]["full_task"]["db"].ravel() for r in results])
    pad = (np.max(all_values) - np.min(all_values)) * .04
    for result in results:
        recording = result["recording"]
        patient, stage, _ = recording["path"].stem.split("_")
        axis = axes[0 if stage == "pre" else 1, int(patient[1]) - 1]
        axis.axhline(0, color="#777777", linewidth=.8, linestyle="--")
        for col, channel in enumerate(CHANNELS):
            for label, hand, shift in [(1, "left", -.16), (-1, "right", .16)]:
                values = result["windows"]["full_task"]["db"][recording["labels"] == label, col]
                jitter = np.random.default_rng(27 + col + label).uniform(-.08, .08, len(values))
                axis.scatter(col + shift + jitter, values, color=colors[hand], s=9, alpha=.5, edgecolors="none")
                axis.plot([col + shift - .10, col + shift + .10], [np.median(values)] * 2,
                          color=colors[hand], linewidth=3)
        axis.set(title=f"{patient} {stage.upper()} · training", xticks=[0, 1], xticklabels=list(CHANNELS),
                 xlim=(-.55, 1.55), ylim=(np.min(all_values) - pad, np.max(all_values) + pad))
        axis.grid(axis="y", color="#EEEEEE", linewidth=.5)
    for axis in axes[:, 0]:
        axis.set_ylabel("Mu-power change (dB)")
    for axis in axes.ravel():
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Same descriptive calculation across six training runs", x=.1, y=.965,
                 ha="left", fontsize=16)
    fig.text(.1, .91, "8–13 Hz | task 2–8 s / baseline −1–0 s | 40 trials per hand in each run", fontsize=11)
    fig.text(.1, .12, "Blue: left trials. Orange: right trials. Dots: all trials. Bars: medians. Axes use the same scale.", fontsize=10)
    fig.text(.1, .075, "Three participants; PRE/POST are first/last rehabilitation sessions. This comparison does not establish recovery.", fontsize=9.5)
    fig.text(.1, .035, "Feedback, artifacts, baseline differences and repeated trials limit interpretation. No test runs used.", fontsize=10, color="#555555")
    paths = {}
    for extension in ("png", "svg"):
        target = Path(output) / f"six_training_runs_mu.{extension}"
        fig.savefig(target, dpi=180, facecolor="white", bbox_inches="tight")
        paths[extension] = str(target)
    plt.close(fig)
    return paths


def run_pass(data_dir=DEFAULT_DATA, output_dir=DEFAULT_OUTPUT):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    estimator_checks = verify_estimator()
    results = [analyze_training(data_dir / f"P{p}_{stage}_training.mat")
               for p in (1, 2, 3) for stage in ("pre", "post")]
    rows = [row for result in results for row in result["rows"]]
    summaries = [row for result in results for row in result["summaries"]]
    write_csv(output_dir / "all_training_trial_mu.csv", rows)
    write_csv(output_dir / "training_mu_summary.csv", summaries)
    write_csv(output_dir / "p1_pre_training_trial_mu.csv", results[0]["rows"])
    # Check export counts and numerical preservation by reading the CSV back.
    with (output_dir / "all_training_trial_mu.csv").open(newline="", encoding="utf-8") as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 960  # 6 runs x 80 trials x 2 channels, not 960 participants.
    assert float(exported[0]["full_task_change_db"]) == rows[0]["full_task_change_db"]
    provenance = {"scope": "Descriptive training runs only; no classifier, no test data.",
                  "primary_run": "P1_pre_training.mat", "channel_columns_1based": {k: v + 1 for k, v in CHANNELS.items()},
                  "channel_source": "Supplied montage.png", "band_hz": BAND_HZ,
                  "baseline_seconds": BASELINE_S, "windows_seconds": WINDOWS,
                  "power_change": "10 * log10(task_mu_power / same_trial_baseline_mu_power)",
                  "welch": {"window": "hann", "nperseg": 256, "noverlap": 128,
                            "nfft": 256, "detrend": "constant", "scaling": "density", "average": "mean"},
                  "segments_per_window": {"baseline": 1, "early": 2, "full_task": 11},
                  "artifact_rejection": "none; all trials shown", "reference": "original supplied reference retained",
                  "physical_units": "unknown; raw power in original units squared; dB is a dimensionless within-trial ratio",
                  "feedback": "Diagram approximately 3.5 s, actual onset not separately recorded; no feedback-free claim",
                  "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
                  "estimator_checks": estimator_checks, "csv_roundtrip_checks": True,
                  "inputs": [{"file": r["recording"]["path"].name, "sha256": r["recording"]["sha256"],
                              "eeg_shape": list(r["recording"]["eeg"].shape), "checks": r["checks"]} for r in results]}
    (output_dir / "provenance_and_checks.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    paths = {"p1": plot_p1(results[0], output_dir), "extension": plot_session_extension(results, output_dir)}
    return {"results": results, "summaries": summaries, "paths": paths, "provenance": provenance}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    bundle = run_pass(args.data_dir, args.output_dir)
    print(json.dumps({"paths": bundle["paths"], "p1_summaries": bundle["results"][0]["summaries"],
                      "checks": bundle["provenance"]["estimator_checks"]}, indent=2))
