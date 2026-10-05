"""Plot calibration-budget sensitivity on the fixed later training trials."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .comparison import MODELS

COLORS = {MODELS[0]: "#BA4F2D", MODELS[1]: "#256691"}
LABELS = {MODELS[0]: "Filter-bank CSP", MODELS[1]: "Riemannian tangent space"}


def plot_forward_calibration(csv_path, png_path):
    """Plot every session in faint lines and the equal-session mean in bold.

    Do not display 120 validation trial decisions as 120 independent patients.
    Error bars are omitted because three participants do not support a precise
    interval for new-patient generalization.
    """
    with Path(csv_path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty forward-calibration summary")
    times = sorted({float(r["decision_time_s"]) for r in rows})
    budgets = sorted({int(r["calibration_trials"]) for r in rows})
    sessions = sorted({(r["patient"], r["session"]) for r in rows})
    if len(times) != 4:
        raise ValueError("The four-panel figure expects four configured decision times")
    lookup = {(r["patient"], r["session"], r["model"],
               float(r["decision_time_s"]), int(r["calibration_trials"])):
              float(r["balanced_accuracy"]) for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate session/model/time/budget combinations")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"}):
        fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), sharex=True, sharey=True)
        fig.subplots_adjust(left=.09, right=.97, top=.79, bottom=.16, wspace=.14, hspace=.27)
        for ax, stop in zip(axes.flat, times):
            for name in MODELS:
                values = np.array([[lookup[p, s, name, stop, n] for n in budgets]
                                   for p, s in sessions])
                for session in values:
                    ax.plot(budgets, 100 * session, color=COLORS[name], lw=.75,
                            alpha=.23, marker="o", markersize=2)
                ax.plot(budgets, 100 * values.mean(axis=0), color=COLORS[name],
                        lw=2.7, marker="o", markersize=5, label=LABELS[name])
            ax.axhline(50, color="#8A8D90", ls=(0, (3, 3)), lw=.8)
            ax.set_title(f"Decision at +{stop:g} s  |  [{stop - 1:g}, {stop:g}) s EEG",
                         loc="left", pad=7, fontsize=10, weight="bold")
            ax.set(ylim=(20, 105), yticks=(25, 50, 75, 100), xticks=budgets)
            ax.grid(axis="y", color="#e9eaed", lw=.6)
        fig.text(.014, .48, "Balanced accuracy on later training trials (%)",
                 rotation=90, va="center", fontsize=10)
        for ax in axes[-1]:
            ax.set_xlabel("Chronological calibration prefix (trials)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.08, .88),
                   ncol=2, frameon=False, fontsize=10)
        fig.suptitle("Calibration and timing on later training trials",
                     x=.09, y=.97, ha="left", fontsize=15, weight="bold")
        fig.text(.09, .915, "Six training runs from three patients  |  Same later trials 61–80 for every point",
                 fontsize=10, color="#555C63")
        fig.text(.09, .08, "Thin lines: individual sessions; thick lines: equal-session mean. Cue at +2 s; feedback timestamps unknown.",
                 fontsize=9, color="#555C63")
        fig.text(.09, .05, "Retrospective, training-only analysis; no newly blind test or inference about rehabilitation benefit.",
                 fontsize=9, color="#555C63")
        path = Path(png_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=320, facecolor="white")
        plt.close(fig)
    return path
