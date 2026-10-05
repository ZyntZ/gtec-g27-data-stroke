"""Validate the animated decision-time figure's saved counts without patient recordings."""
import csv
from pathlib import Path

import pytest

from analysis.decision_clock import (HOLDS, SESSIONS, STATIC_POOLED, TIMES, frame_times,
                                     read_counts, render)

ROOT = Path(__file__).resolve().parents[1]
COUNTS = ROOT / "results" / "decision_clock" / "decision_clock_counts.csv"


def test_saved_counts_cover_the_grid_and_agree_with_the_static_figure():
    counts = read_counts(COUNTS)
    assert len(counts) == len(SESSIONS) * len(TIMES) == 174
    with (ROOT / "results/jury_tradeoff/same_tail_counts.csv").open(newline="") as stream:
        static = {(r["patient"], r["session"], float(r["decision_time_s"])): int(r["correct"])
                  for r in csv.DictReader(stream) if r["patient"] != "pooled"}
    assert set(STATIC_POOLED) == {key[2] for key in static}
    for key, correct in static.items():  # per session, not only pooled
        assert counts[key] == correct


def test_incomplete_or_inconsistent_counts_are_rejected(tmp_path):
    rows = list(csv.DictReader(COUNTS.open(newline="")))
    def write(changed):
        path = tmp_path / "counts.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(changed)
        return path
    with pytest.raises(ValueError, match="every session"):
        read_counts(write(rows[:-1]))
    bumped = [dict(r) for r in rows]
    target = next(r for r in bumped if float(r["decision_time_s"]) == 3.5 and int(r["correct"]) < 20)
    target["correct"] = int(target["correct"]) + 1
    with pytest.raises(ValueError, match="static figure"):
        read_counts(write(bumped))


def test_frames_are_chronological_pause_on_named_endpoints_and_render(tmp_path):
    frames = frame_times()
    assert frames == sorted(frames) and frames[0] == TIMES[0] and frames[-1] == TIMES[-1]
    assert all(frames.count(time) == 1 + extra for time, extra in HOLDS.items())
    output = render(read_counts(COUNTS), tmp_path / "clock.gif", dpi=40)
    assert output.stat().st_size > 10_000 and output.with_suffix(".png").is_file()
