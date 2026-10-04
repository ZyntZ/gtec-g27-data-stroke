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


THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df",
                  band="#efeeea", csp="#eb6834", riemann="#2a78d6"),
    "dark": dict(surface="#000000", ink="#ffffff", ink2="#c3c2b7", grid="#2c2c2a",
                 band="#1f1f1d", csp="#d95926", riemann="#3987e5"),
}


def plot_decision_time(csv_path, png_path, *, theme="light", cue_s=2.0, feedback_s=3.5):
    """Held-out accuracy against decision time: pooled, then one panel per session."""
    with Path(csv_path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    c = THEMES[theme]
    series = (("filterbank_riemann_recentered", "Riemannian + recentering", c["riemann"]),
              ("filterbank_csp_shrinkage_lda", "Filter-bank CSP + LDA", c["csp"]))
    times = sorted({float(r["decision_time_s"]) for r in rows})

    def curve(model, session=None):
        picked = [r for r in rows if r["model"] == model and
                  (session is None or (r["patient"], r["session"]) == session)]
        return [100 * sum(int(r["correct_test_trials"]) for r in picked
                          if float(r["decision_time_s"]) == t)
                / sum(int(r["test_trials"]) for r in picked
                      if float(r["decision_time_s"]) == t) for t in times]

    def dress(ax, small=False):
        ax.set_facecolor(c["surface"])
        ax.axvspan(feedback_s, 8, color=c["band"], linewidth=0, zorder=0)
        ax.axvline(cue_s, color=c["ink2"], linewidth=1)
        ax.axhline(50, color=c["ink2"], linewidth=1, linestyle=(0, (4, 3)))
        ax.set(xlim=(1, 8), ylim=(35, 100), xticks=range(1, 9),
               yticks=(50, 75, 100) if small else (40, 50, 60, 70, 80, 90, 100))
        ax.grid(axis="y", color=c["grid"], linewidth=1)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(c["grid"])
        ax.tick_params(colors=c["ink2"], length=0, labelsize=8 if small else 10)

    fig = plt.figure(figsize=(10, 7.4), facecolor=c["surface"])
    grid = fig.add_gridspec(2, 6, height_ratios=(3.1, 1), hspace=0.42, wspace=0.28,
                            left=0.07, right=0.98, top=0.86, bottom=0.08)
    main = fig.add_subplot(grid[0, :])
    dress(main)
    for model, label, color in series:
        main.plot(times, curve(model), color=color, linewidth=2, label=label,
                  solid_capstyle="round")
    main.set_ylabel("Held-out test trials correct (%), 480 trials", color=c["ink2"])
    main.set_xlabel("Decision time after trial trigger (s): end of a causal 1 s window",
                    color=c["ink2"])
    for x, text in ((1.06, "before the\ninstruction"), (cue_s + 0.06, "early post-cue"),
                    (feedback_s + 0.06, "feedback phase: visual + electrical stimulation\n"
                     "may be on (protocol timing; not recorded per trial)")):
        main.text(x, 98.5, text, color=c["ink2"], fontsize=9, va="top", linespacing=1.3)
    main.text(7.94, 51, "chance", color=c["ink2"], fontsize=9, ha="right", va="bottom")
    main.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=c["ink"],
                bbox_to_anchor=(1.0, 0.04))
    fig.text(0.07, 0.955, "Decoding rises only after the cue and peaks once feedback can be present",
             color=c["ink"], fontsize=14, fontweight="bold")
    fig.text(0.07, 0.915, "Each model is fitted on the training run and scored once per test trial; "
             "causal filters, so no EEG after the decision time is used.",
             color=c["ink2"], fontsize=9.5)
    for i, session in enumerate((p, s) for p in ("P1", "P2", "P3") for s in ("pre", "post")):
        ax = fig.add_subplot(grid[1, i])
        dress(ax, small=True)
        for model, _, color in series:
            ax.plot(times, curve(model, session), color=color, linewidth=1.5)
        ax.set_title(f"{session[0]} {session[1].upper()}", color=c["ink"], fontsize=10,
                     loc="left", pad=4)
        ax.set_xticks((2, 3.5, 8), labels=("2", "3.5", "8"))
        if i:
            ax.set_yticklabels([])
    path = Path(png_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=260, facecolor=c["surface"])
    plt.close(fig)
    return path
