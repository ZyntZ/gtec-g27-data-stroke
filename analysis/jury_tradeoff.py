"""Rebuild a training-only, decision-latency trade-off figure from saved trials.

Usage: python analysis/jury_tradeoff.py --input results/forward_calibration/forward_calibration_predictions.csv
The dataset is not required to draw this figure; regenerate the input with
`stroke-rehab forward-calibration` after verifying the organizer archive.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODEL = "filterbank_csp_shrinkage_lda"
TIMES = (2.0, 3.5, 4.25, 6.5)
SESSIONS = tuple((f"P{patient}", stage)
                 for patient in (1, 2, 3) for stage in ("pre", "post"))
TRIALS = range(61, 81)


def read_same_tail(path):
    """Require identical trials and labels at every plotted decision time."""
    predictions = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["model"] != MODEL or int(row["calibration_trials"]) != 60:
                continue
            pair = (row["patient"], row["session"])
            time = float(row["decision_time_s"])
            trial = int(row["validation_trial_1based"])
            key = (*pair, time, trial)
            if key in predictions:
                raise ValueError(f"Duplicate prediction: {key}")
            true, pred, correct = (int(row[field]) for field in
                                   ("true_label", "predicted_label", "correct"))
            if pair not in SESSIONS or time not in TIMES or trial not in TRIALS:
                raise ValueError(f"Unexpected session/time/trial: {key}")
            if true not in (-1, 1) or pred not in (-1, 1) or correct != int(true == pred):
                raise ValueError(f"Invalid labels or correctness: {key}")
            predictions[key] = (true, correct)
    expected = {(*pair, time, trial) for pair in SESSIONS
                for time in TIMES for trial in TRIALS}
    if set(predictions) != expected:
        raise ValueError(f"Incomplete/mismatched tail: missing {len(expected - set(predictions))}, "
                         f"extra {len(set(predictions) - expected)}")
    counts = defaultdict(int)
    for pair in SESSIONS:
        for trial in TRIALS:
            labels = {predictions[(*pair, time, trial)][0] for time in TIMES}
            if len(labels) != 1:
                raise ValueError(f"Labels differ between times: {(*pair, trial)}")
            for time in TIMES:
                counts[(*pair, time)] += predictions[(*pair, time, trial)][1]
    return counts


def render(counts, input_path, output_dir):
    """Render a fixed, descriptive comparison; no test runs or time selection."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
    rows = []
    for pair in SESSIONS:
        for time in TIMES:
            rows.append(dict(patient=pair[0], session=pair[1], decision_time_s=time,
                             correct=counts[(*pair, time)], trials=20,
                             accuracy=counts[(*pair, time)] / 20,
                             source_sha256=source_hash))
    for time in TIMES:
        total = sum(counts[(*pair, time)] for pair in SESSIONS)
        rows.append(dict(patient="pooled", session="six runs / three people",
                         decision_time_s=time, correct=total, trials=120,
                         accuracy=total / 120, source_sha256=source_hash))
    out_csv = output_dir / "same_tail_counts.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    fig, (ax, pooled) = plt.subplots(1, 2, figsize=(11.8, 5.8),
                                     gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(left=.065, right=.96, top=.77, bottom=.24, wspace=.31)
    ink, early, late = "#213548", "#007C83", "#CA5E39"
    xs = np.arange(len(SESSIONS))
    ax.bar(xs - .18, [counts[(*pair, 3.5)] / 20 for pair in SESSIONS],
           width=.34, color=early, label="+3.5 s / fixed early")
    ax.bar(xs + .18, [counts[(*pair, 4.25)] / 20 for pair in SESSIONS],
           width=.34, color=late, label="+4.25 s / later EEG")
    ax.set_xticks(xs, [f"{p} {s.upper()}" for p, s in SESSIONS], rotation=30,
                  ha="right")
    ax.set_ylim(0, 1.09)
    ax.set_ylabel("Correct / 20 later training trials")
    ax.axhline(.5, color="#A5ADB4", linewidth=.8, linestyle=":")
    ax.set_title("Same 20 trial IDs per session", loc="left", color=ink, pad=14)
    ax.legend(frameon=False, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.06),
              fontsize=8.3)
    ax.grid(axis="y", color="#E6EBED", linewidth=.6)
    ax.set_axisbelow(True)

    pooled_values = np.array([sum(counts[(*pair, t)] for pair in SESSIONS)
                              for t in TIMES])
    pooled.plot(TIMES, pooled_values / 120, color=ink, marker="o", linewidth=1.8)
    pooled.scatter([3.5, 4.25], pooled_values[[1, 2]] / 120, color=[early, late],
                   zorder=3, s=74)
    for t, n in zip(TIMES, pooled_values):
        pooled.annotate(f"{n}/120", (t, n / 120), xytext=(5, 9),
                        textcoords="offset points", color=ink, fontsize=9)
    pooled.axhline(.5, color="#A5ADB4", linewidth=.8, linestyle=":")
    pooled.axvline(3.5, color=early, linewidth=.8, linestyle="--", alpha=.7)
    pooled.set(xlim=(1.65, 6.95), ylim=(.37, .98), xticks=TIMES,
               xlabel="EEG window end (s after trigger)", ylabel="Pooled accuracy")
    pooled.set_title("More time, more decodable signal", loc="left", color=ink, pad=14)
    pooled.grid(axis="y", color="#E6EBED", linewidth=.6)
    pooled.set_axisbelow(True)
    fig.suptitle("Later accuracy is not cleaner motor-intent evidence", x=.065,
                 ha="left", y=.96, fontsize=18, fontweight="bold", color=ink)
    fig.text(.065, .87, "Training only · 60 chronological calibration trials · fixed CSP · "
             "trials 61–80 · n=3 patients / 6 sessions", fontsize=10, color="#4E5E6C")
    fig.text(.065, .055, "Training contains feedback; 3.5 s is a diagram-based boundary, "
             "not a measured FES onset. Retrospective diagnostic, not a clinical outcome.",
             fontsize=9, color="#4E5E6C")
    fig.savefig(output_dir / "same_tail_tradeoff.png", dpi=300)
    fig.savefig(output_dir / "same_tail_tradeoff.svg")
    plt.close(fig)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/forward_calibration/forward_calibration_predictions.csv")
    parser.add_argument("--output-dir", default="results/jury_tradeoff")
    args = parser.parse_args()
    rows = render(read_same_tail(args.input), args.input, args.output_dir)
    print("Training-only pooled counts:", {row["decision_time_s"]: row["correct"]
                                          for row in rows if row["patient"] == "pooled"})


if __name__ == "__main__":
    main()
