"""Compact per-session summary; never imply these three patients are a clinical trial."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_results(csv_path, png_path):
    with Path(csv_path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    patients = ("P1", "P2", "P3")
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    x = np.arange(len(patients))
    width = 0.32
    colors = {"pre": "#39739D", "post": "#DC785C"}
    for stage, displacement in (("pre", -width / 2), ("post", width / 2)):
        subset = [next(r for r in rows if r["patient"] == p and
                       r["session"] == stage) for p in patients]
        values = [100 * float(r["test_accuracy"]) for r in subset]
        bars = ax.bar(x + displacement, values, width, label=stage.upper(),
                      color=colors[stage], edgecolor="white", linewidth=0.6)
        for bar, row in zip(bars, subset):
            ax.annotate(f'{row["correct_test_trials"]}/{row["test_trials"]}',
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=8)
    ax.axhline(50, color="#4D535A", linewidth=1, linestyle="--", label="Chance (balanced)")
    ax.set(ylim=(0, 108), xticks=x, xticklabels=patients,
           ylabel="Held-out test trials correct (%)",
           title="Motor-imagery decoding by rehabilitation session")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=8)
    ax.text(0, -0.27, "Each model is selected by training-run CV; test labels are used only to score.",
            transform=ax.transAxes, fontsize=8, color="#50575E")
    path = Path(png_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
