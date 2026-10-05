"""Animate the README's training-only decision-time trade-off as a sweeping trial clock.

Same design as `analysis/jury_tradeoff.py`: fixed CSP, the first 60 training
trials calibrate, trials 61-80 of each training run are scored. Here the
decision time advances in 0.25 s steps so the curve can be drawn as it grows.

  python analysis/decision_clock.py --data-dir data/stroke-rehab   # recompute counts, then draw
  python analysis/decision_clock.py                                # draw from the saved counts

Only training recordings are read. Drawing needs no EEG data.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

MODEL = "filterbank_csp_shrinkage_lda"
CALIBRATION_TRIALS = 60
TIMES = tuple(float(t) for t in np.arange(1.0, 8.0 + 1e-9, 0.25))
SESSIONS = tuple((f"P{patient}", stage) for patient in (1, 2, 3) for stage in ("pre", "post"))
# The four endpoints of the static figure; their pooled counts must match it.
STATIC_POOLED = {2.0: 55, 3.5: 82, 4.25: 99, 6.5: 99}
INSTRUCTION_S, ASSUMED_FEEDBACK_S = 2.0, 3.5
HOLDS = {3.5: 14, 4.25: 14, 8.0: 22}  # extra frames to pause on
DEFAULT_OUTPUT = Path("results/decision_clock")


def compute_counts(data_dir, output_csv):
    """Score trials 61-80 of every TRAINING run at each decision time."""
    from stroke_rehab.data import read_recording
    from stroke_rehab.forward_calibration import score_recording, training_files

    rows = []
    for patient, stage, path in training_files(data_dir):
        predictions = score_recording(read_recording(path), patient, stage,
                                      budgets=(CALIBRATION_TRIALS,), stops=TIMES)
        for time in TIMES:
            correct = sum(r["correct"] for r in predictions
                          if r["model"] == MODEL and r["decision_time_s"] == time)
            rows.append(dict(patient=patient, session=stage, decision_time_s=time,
                             correct=correct, trials=20))
        print(f"{patient} {stage}: {len(TIMES)} decision times scored", flush=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_csv


def read_counts(path):
    """Return {(patient, session, time): correct}; require a complete, consistent grid."""
    counts = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["patient"], row["session"], float(row["decision_time_s"]))
            if key in counts or int(row["trials"]) != 20 or not 0 <= int(row["correct"]) <= 20:
                raise ValueError(f"Duplicate or invalid row: {key}")
            counts[key] = int(row["correct"])
    expected = {(*pair, time) for pair in SESSIONS for time in TIMES}
    if set(counts) != expected:
        raise ValueError("Counts do not cover every session and decision time exactly once")
    for time, total in STATIC_POOLED.items():
        if sum(counts[(*pair, time)] for pair in SESSIONS) != total:
            raise ValueError(f"Pooled count at {time} s disagrees with the static figure")
    return counts


def frame_times():
    """Decision time shown in each frame, pausing on the endpoints named in the README."""
    frames = []
    for time in TIMES:
        frames.extend([time] * (1 + HOLDS.get(time, 0)))
    return frames


def render(counts, output_gif, *, fps=8, dpi=110):
    ink, muted, early, late = "#213548", "#4E5E6C", "#007C83", "#CA5E39"
    grid, band, surface = "#E6EBED", "#F1F3F4", "#FFFFFF"
    times = np.array(TIMES)
    pooled = np.array([sum(counts[(*pair, t)] for pair in SESSIONS) for t in TIMES])
    session = {pair: np.array([counts[(*pair, t)] for t in TIMES]) * 5.0 for pair in SESSIONS}

    fig = plt.figure(figsize=(8.6, 5.4), facecolor=surface)
    fig.text(.07, .935, "Accuracy depends on when the decoder has to decide",
             fontsize=15, fontweight="bold", color=ink)
    fig.text(.07, .885, "Training runs only · fixed CSP · 60 calibration trials · the same later "
             "20 trials per session · 3 people / 6 sessions", fontsize=8.6, color=muted)
    clock = fig.add_axes([.07, .775, .9, .055])
    ax = fig.add_axes([.07, .2, .9, .5])
    clock.set(xlim=(0, 8), ylim=(0, 1))
    clock.axis("off")
    for start, stop, colour in ((0, INSTRUCTION_S, "#DDE2E5"), (INSTRUCTION_S, ASSUMED_FEEDBACK_S, "#9FD3D6"),
                                (ASSUMED_FEEDBACK_S, 8, "#EBC7B8")):
        clock.barh(.5, stop - start - .02, .55, left=start + .01, color=colour)
    for x, label in ((.06, "get ready"), (INSTRUCTION_S + .06, "instructed imagery"),
                     (ASSUMED_FEEDBACK_S + .06, "feedback may be on (start assumed, not recorded)")):
        clock.text(x, .5, label, va="center", fontsize=8.2, color=ink)
    cursor = clock.plot([1, 1], [-.1, 1.1], color=ink, linewidth=2.2, clip_on=False)[0]

    ax.set_facecolor(surface)
    ax.axvspan(ASSUMED_FEEDBACK_S, 8, color=band, linewidth=0, zorder=0)
    ax.axvline(INSTRUCTION_S, color=muted, linewidth=1)
    ax.axvline(ASSUMED_FEEDBACK_S, color=muted, linewidth=1, linestyle=(0, (1, 2.5)))
    ax.axhline(50, color="#A5ADB4", linewidth=.9, linestyle=":")
    ax.text(7.95, 51.5, "chance", ha="right", fontsize=8, color=muted)
    ax.text(INSTRUCTION_S + .06, 20.8, "instruction (documented)", fontsize=7.6, color=muted, va="bottom")
    ax.text(ASSUMED_FEEDBACK_S + .06, 20.8, "feedback start (assumed)", fontsize=7.6, color=muted, va="bottom")
    ax.set(xlim=(0, 8), ylim=(20, 102), xticks=range(0, 9), yticks=(30, 40, 50, 60, 70, 80, 90, 100))
    ax.set_xlabel("Decision time after trial trigger (s): end of a causal 1 s window",
                  fontsize=9, color=muted)
    ax.set_ylabel("Later training trials correct (%)", fontsize=9, color=muted)
    ax.grid(axis="y", color=grid, linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(grid)
    ax.tick_params(colors=muted, length=0, labelsize=8.5)
    thin = {pair: ax.plot([], [], color="#C9D1D7", linewidth=.9)[0] for pair in SESSIONS}
    line = ax.plot([], [], color=ink, linewidth=2.4, solid_capstyle="round")[0]
    head = ax.plot([], [], "o", color=ink, markersize=8, markeredgecolor=surface, markeredgewidth=2)[0]
    marks = {3.5: ax.plot([], [], "o", color=early, markersize=9, markeredgecolor=surface,
                          markeredgewidth=2, zorder=5)[0],
             4.25: ax.plot([], [], "o", color=late, markersize=9, markeredgecolor=surface,
                           markeredgewidth=2, zorder=5)[0]}
    notes = {3.5: ax.text(3.42, 0, "", ha="right", fontsize=9, color=ink, fontweight="bold"),
             4.25: ax.text(4.4, 0, "", ha="left", fontsize=9, color=ink, fontweight="bold")}
    counter = ax.text(.12, 99, "", va="top", fontsize=15, fontweight="bold", color=ink,
                      linespacing=1.25)
    fig.text(.07, .035, "Grey lines: single sessions. Feedback is present in training runs; +3.5 s comes "
             "from the protocol diagram, not a measured stimulation onset.\nRetrospective training-only "
             "diagnostic on three people: not a test score, a clinical outcome or evidence of recovery.",
             fontsize=7.6, color=muted, linespacing=1.45)

    def draw(now):
        shown = times <= now + 1e-9
        cursor.set_xdata([now, now])
        line.set_data(times[shown], pooled[shown] / 1.2)
        head.set_data([now], [pooled[shown][-1] / 1.2])
        for pair, artist in thin.items():
            artist.set_data(times[shown], session[pair][shown])
        for time, artist in marks.items():
            if now + 1e-9 >= time:
                value = pooled[TIMES.index(time)]
                artist.set_data([time], [value / 1.2])
                notes[time].set_position((notes[time].get_position()[0], value / 1.2 + 3))
                notes[time].set_text(f"{value}/120 at +{time:g} s")
            else:
                artist.set_data([], [])
                notes[time].set_text("")
        counter.set_text(f"+{now:.2f} s\n{pooled[shown][-1]}/120 correct")
        return ()

    output_gif = Path(output_gif)
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, draw, frames=frame_times(), blit=False).save(
        output_gif, writer=PillowWriter(fps=fps), dpi=dpi)
    draw(TIMES[-1])
    fig.savefig(output_gif.with_suffix(".png"), dpi=200)  # static fallback: the final frame
    plt.close(fig)
    return output_gif


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None, help="Recompute counts from training recordings")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT, type=Path)
    args = parser.parse_args()
    counts_csv = args.output_dir / "decision_clock_counts.csv"
    if args.data_dir is not None:
        compute_counts(args.data_dir, counts_csv)
    print(render(read_counts(counts_csv), args.output_dir / "decision_clock.gif"))
