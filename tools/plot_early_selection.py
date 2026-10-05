"""Rebuild the descriptive six-session figure from committed summary rows."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/early_selection"
STYLES = (
    ("filterbank_csp_shrinkage_lda", "CSP + shrinkage LDA", "#0072B2", "o", "-"),
    ("filterbank_riemann_recentered", "Adaptive Riemannian", "#D55E00", "s", "-"),
    ("training_selected", "Training-selected policy", "#222222", "D", "--"),
    ("calibration_majority", "Calibration majority", "#777777", None, ":"),
)


def main():
    with (OUT / "early_summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans",
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True, sharey=True)
    for p, patient in enumerate(("P1", "P2", "P3")):
        for s, session in enumerate(("pre", "post")):
            ax = axes[p, s]
            for strategy, label, color, marker, style in STYLES:
                cell = sorted((r for r in rows if (r["patient"], r["session"], r["strategy"])
                               == (patient, session, strategy)), key=lambda r: int(r["calibration_trials"]))
                assert len(cell) == 4 and all(int(r["trials"]) == 20 for r in cell)
                ax.plot([int(r["calibration_trials"]) for r in cell],
                        [float(r["balanced_accuracy"]) for r in cell],
                        color=color, marker=marker, linestyle=style, label=label,
                        linewidth=1.6, markersize=7 if strategy == "training_selected" else 5,
                        markerfacecolor="none" if strategy == "training_selected" else color,
                        zorder=4 if strategy == "training_selected" else 2)
            ax.set_title(f"{patient}  |  {session.upper()}", loc="left", weight="bold")
            ax.set_ylim(0, 1.03)
            ax.set_yticks([0, .25, .5, .75, 1])
            ax.set_xticks([10, 20, 40, 60])
            ax.grid(axis="y", color="#dddddd", linewidth=.6)
            if s == 0:
                ax.set_ylabel("Balanced accuracy")
            if p == 2:
                ax.set_xlabel("Calibration trials (chronological prefix)")
    fig.suptitle("Fixed +3.5 s decisions after chronological calibration", fontsize=15, y=.98)
    fig.text(.5, .941, "Shared tail: trials 61–80 · 20 trials per panel / budget · three participants, six sessions",
             ha="center", fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .06), ncol=2, frameon=False)
    selected = sum(int(r["correct"]) for r in rows if r["strategy"] == "training_selected" and r["calibration_trials"] == "60")
    csp = sum(int(r["correct"]) for r in rows if r["strategy"] == STYLES[0][0] and r["calibration_trials"] == "60")
    fig.text(.5, .031, f"At 60 calibration trials: CSP {csp}/120 correct; selected {selected}/120. No pooled accuracy gain.",
             ha="center", fontsize=10)
    fig.text(.5, .011, "Retrospective descriptive audit: these tails were already inspected; sessions are repeated within participants.",
             ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(left=.09, right=.985, bottom=.18, top=.895, hspace=.34, wspace=.12)
    for suffix in ("svg", "png"):
        fig.savefig(OUT / f"early_calibration.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
