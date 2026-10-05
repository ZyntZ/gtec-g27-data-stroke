"""Organizer-style peak-of-mean accuracy, with an honest fixed-time comparator.

This is exploratory: the supplied test MAT files have already been inspected.
For each endpoint one CSP pipeline is fit on all 80 training trials and predicts
all 80 test trials. The peak is chosen ONLY while summarizing test accuracy;
there is no claim that this post-hoc choice is deployable or blinded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import sklearn

from .comparison import MODELS, _design, _estimator
from .data import read_recording, session_paths
from .features import BANDS
from .riemann import filtered_trials

# A one-second causal window ending at each point, including the feedback phase.
MODEL = MODELS[0]
WINDOW_S = 1.0
STOPS_S = tuple(round(i / 4, 2) for i in range(10, 33))  # 2.50 ... 8.00 s
FIXED_EARLY_S = 3.5


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source_digest(path):
    """Fingerprint source consistently across Git's Windows/Linux checkouts."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def peak_of_mean(correct, stops):
    """Max_t(mean_trial(correct[t, trial])); earliest endpoint wins exact ties.

    The array must be a complete rectangular grid of observed binary outcomes.
    Taking max_time first *per trial* would answer a different, oracle question.
    """
    values = np.asarray(correct)
    times = np.asarray(stops, dtype=float)
    if (values.ndim != 2 or not all(s > 0 for s in values.shape)
            or len(times) != values.shape[0] or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
            or not np.isin(values, (0, 1)).all()):
        raise ValueError("Expected a complete binary time-by-trial grid with increasing endpoints")
    counts = values.astype(np.int64).sum(axis=1)
    peak_index = int(np.argmax(counts))
    return peak_index, int(counts[peak_index]), float(counts[peak_index] / values.shape[1])


def score_recording(train, test, patient, session, *, stops=STOPS_S,
                    fixed_s=FIXED_EARLY_S):
    """Train on calibration labels only; score one common 80-trial test grid.

    Each time point gets its own fit. The exact same fitting and preprocessing
    rule is used at every time; there is no tuning from test labels. Test labels
    are accessed only after the entire prediction matrix has been assembled.
    """
    stops = tuple(float(t) for t in stops)
    if (len(stops) == 0 or not np.isfinite(stops).all()
            or np.any(np.diff(stops) <= 0)
            or stops[0] - WINDOW_S < 0 or stops[-1] > 8
            or float(fixed_s) not in stops):
        raise ValueError("Endpoints must increase within one trial and include the fixed time")
    if (train.fs != test.fs or train.fs != 256 or train.signal.shape[1] != test.signal.shape[1]
            or len(train.labels) != 80 or len(test.labels) != 80):
        raise ValueError("Expected matching 80-trial, 256 Hz training and test runs")
    # Keep test labels out of fitting and feature extraction. Band filters are
    # forward-only on continuous runs; different endpoints share filter state.
    train_epochs = filtered_trials(train, BANDS, causal=True)
    test_epochs = filtered_trials(test, BANDS, causal=True)
    predicted = np.empty((len(stops), len(test.onsets)), dtype=np.int8)
    for idx, stop in enumerate(stops):
        window = (stop - WINDOW_S, stop)
        x_train = _design(MODEL, train_epochs, train.fs, window)
        x_test = _design(MODEL, test_epochs, test.fs, window)
        classifier = _estimator(MODEL).fit(x_train, train.labels)
        predicted[idx] = classifier.predict(x_test)
    truth = np.asarray(test.labels)
    if not np.isin(truth, (-1, 1)).all() or not np.isin(predicted, (-1, 1)).all():
        raise ValueError("Only complete left/right predictions and labels are allowed")
    correct = predicted == truth[None, :]
    best, count, value = peak_of_mean(correct, stops)
    fixed = stops.index(float(fixed_s))
    rows = []
    for idx, stop in enumerate(stops):
        left = truth == 1
        right = truth == -1
        rows.append(dict(patient=patient, session=session, model=MODEL,
                         window_start_s=stop - WINDOW_S, decision_time_s=stop,
                         correct=int(correct[idx].sum()), trials=len(truth),
                         accuracy=float(correct[idx].mean()),
                         left_correct=int(correct[idx, left].sum()),
                         left_trials=int(left.sum()),
                         right_correct=int(correct[idx, right].sum()),
                         right_trials=int(right.sum()),
                         balanced_accuracy=float((correct[idx, left].mean() +
                                                  correct[idx, right].mean()) / 2)))
    summary = dict(patient=patient, session=session, model=MODEL,
                   calibration_trials=len(train.labels), test_trials=len(truth),
                   grid_points=len(stops), fixed_time_s=stops[fixed],
                   fixed_correct=rows[fixed]["correct"],
                   fixed_accuracy=rows[fixed]["accuracy"],
                   peak_time_s=stops[best], peak_correct=count,
                   peak_accuracy=value,
                   peak_minus_fixed_correct=count - rows[fixed]["correct"],
                   peak_is_posthoc=True)
    return rows, summary


def aggregate_session_scores(time_rows, session_summaries):
    """Distinguish pooled peak from the sum of patient/session-wise peaks.

    The pooled curve first sums matched trial counts across sessions at *each*
    timestamp and takes one maximum; session peaks independently choose their
    timestamps. Both post-hoc operations use test labels and are exploratory.
    """
    if not time_rows or not session_summaries:
        raise ValueError("Missing timecourse or session summary")
    keys = {(r["patient"], r["session"]) for r in session_summaries}
    if len(keys) != len(session_summaries):
        raise ValueError("Duplicate session summary")
    by_session = {}
    for row in time_rows:
        key = (row["patient"], row["session"])
        if key not in keys:
            raise ValueError("Timecourse does not match session summaries")
        by_session.setdefault(key, []).append(row)
    if set(by_session) != keys:
        raise ValueError("A session is missing from the timecourse")
    first = sorted(keys)[0]
    times = [float(r["decision_time_s"]) for r in by_session[first]]
    for key, rows in by_session.items():
        if len(rows) != len(times) or [float(r["decision_time_s"]) for r in rows] != times:
            raise ValueError("Each session must have the same ordered time grid")
        if any(int(r["trials"]) != 80 for r in rows):
            raise ValueError("Unexpected test trial count")
    counts = np.array([[int(row["correct"]) for row in by_session[key]]
                       for key in sorted(keys)])
    if np.any((counts < 0) | (counts > 80)):
        raise ValueError("Accuracy counts must lie within 0..80")
    pooled = counts.sum(axis=0)
    peak = int(np.argmax(pooled))  # earliest timestamp wins a tie.
    fixed_times = {float(r["fixed_time_s"]) for r in session_summaries}
    if len(fixed_times) != 1 or next(iter(fixed_times)) not in times:
        raise ValueError("Session summaries disagree about fixed decision time")
    fixed = times.index(next(iter(fixed_times)))
    for i, key in enumerate(sorted(keys)):
        row = next(s for s in session_summaries
                   if (s["patient"], s["session"]) == key)
        if (int(row["test_trials"]) != 80 or
                int(row["fixed_correct"]) != int(counts[i, fixed]) or
                int(row["peak_correct"]) != int(counts[i].max())):
            raise ValueError("Summary disagrees with its session timecourse")
    n_trials = int(sum(int(s["test_trials"]) for s in session_summaries))
    return dict(sessions=len(keys), test_trials=n_trials, grid_points=len(times),
                fixed_time_s=times[fixed], fixed_correct=int(pooled[fixed]),
                fixed_accuracy=float(pooled[fixed] / n_trials),
                pooled_peak_time_s=times[peak], pooled_peak_correct=int(pooled[peak]),
                pooled_peak_accuracy=float(pooled[peak] / n_trials),
                sum_of_session_peaks_correct=sum(int(s["peak_correct"]) for s in session_summaries),
                sum_of_session_peaks_accuracy=float(sum(int(s["peak_correct"])
                                                    for s in session_summaries) / n_trials),
                all_peaks_posthoc=True)


def plot_timecourse(time_rows, session_summaries, output_file):
    """Save six panels with a common scale and explicit retrospective peaks."""
    import matplotlib
    import matplotlib.pyplot as plt

    if len(session_summaries) != 6 or len(time_rows) != 6 * len(STOPS_S):
        raise ValueError("Expected complete six-session timecourse for this figure")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.75, "figure.facecolor": "white"}):
        fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.3), sharex=True, sharey=True)
        for row, session in enumerate(("pre", "post")):
            for col, patient in enumerate(("P1", "P2", "P3")):
                ax = axes[row, col]
                values = sorted((r for r in time_rows
                                 if r["patient"] == patient and r["session"] == session),
                                key=lambda r: r["decision_time_s"])
                match = [s for s in session_summaries
                         if s["patient"] == patient and s["session"] == session]
                if len(match) != 1 or len(values) != len(STOPS_S):
                    raise ValueError("Incomplete session for figure")
                entry = match[0]
                x = np.array([r["decision_time_s"] for r in values])
                y = np.array([r["accuracy"] for r in values])
                ax.plot(x, y, lw=1.6, color="#283747", marker="o", ms=2.7)
                ax.scatter([entry["fixed_time_s"]], [entry["fixed_accuracy"]],
                           s=39, color="#007E83", zorder=4)
                ax.scatter([entry["peak_time_s"]], [entry["peak_accuracy"]],
                           s=52, marker="D", color="#BC4B30", zorder=5)
                ax.axhline(0.5, lw=0.7, color="#A8ACB1", ls=":", zorder=0)
                ax.set(title=f"{patient} · {session.upper()}", ylim=(0.33, 1.04),
                       xlim=(2.4, 8.1), xticks=(2.5, 3.5, 4.5, 5.5, 6.5, 7.5))
                ax.tick_params(direction="out", length=3)
                if col == 0:
                    ax.set_ylabel("Trial accuracy")
                if row == 1:
                    ax.set_xlabel("Seconds after trigger")
                ax.text(0.98, 0.05,
                        f"early {entry['fixed_correct']}/80  ·  peak {entry['peak_correct']}/80",
                        transform=ax.transAxes, fontsize=7, ha="right", va="bottom")
        handles = [plt.Line2D([], [], color="#283747", lw=1.6, marker="o", ms=3),
                   plt.Line2D([], [], color="#007E83", marker="o", ls="None", ms=6),
                   plt.Line2D([], [], color="#BC4B30", marker="D", ls="None", ms=6)]
        fig.legend(handles, ["Causal 1-s windows", "Fixed 3.5 s", "Test-selected peak"],
                   loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.985))
        fig.suptitle("Accuracy over decision time", y=1.045, fontsize=12, weight="bold")
        fig.text(0.5, 0.01,
                 "Exploratory: six previously inspected test runs; peaks are retrospective, not online decisions.",
                 ha="center", fontsize=8, color="#525B66")
        fig.tight_layout(rect=(0.02, 0.045, 1, 0.91), w_pad=1.4, h_pad=1.1)
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=330, facecolor="white", bbox_inches="tight")
        plt.close(fig)
    return path


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _check_reference(rows, reference_csv):
    """Optional parity gate against the older, same-model causal time sweep."""
    with Path(reference_csv).open(newline="", encoding="utf-8") as stream:
        old = list(csv.DictReader(stream))
    keys = [(r["patient"], r["session"], r["model"], float(r["decision_time_s"]))
            for r in old]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate prior timecourse key")
    reference = dict(zip(keys, old))
    for row in rows:
        key = (row["patient"], row["session"], row["model"], row["decision_time_s"])
        previous = reference.get(key)
        if (previous is None or int(previous["correct_test_trials"]) != row["correct"]
                or int(previous["test_trials"]) != row["trials"]
                or float(previous["window_s"]) != WINDOW_S):
            raise ValueError(f"Historical decision-time mismatch at {key}")


def run(data_root, output_dir, *, reference_csv=None, stops=STOPS_S,
        fixed_s=FIXED_EARLY_S, make_plot=True):
    """Save aggregate accuracies and an input/source receipt, not raw EEG."""
    records, summaries, hashes = [], [], {}
    for patient, session, train_file, test_file in session_paths(data_root):
        training, testing = read_recording(train_file), read_recording(test_file)
        hashes[train_file.name] = _digest(train_file)
        hashes[test_file.name] = _digest(test_file)
        per_time, result = score_recording(training, testing, patient, session,
                                           stops=stops, fixed_s=fixed_s)
        records.extend(per_time)
        summaries.append(result)
    if reference_csv is not None:
        _check_reference(records, reference_csv)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "organizer_metric_timecourse.csv", records)
    _write_csv(out / "organizer_metric_summary.csv", summaries)
    _write_csv(out / "organizer_metric_aggregate.csv",
               [aggregate_session_scores(records, summaries)])
    if make_plot and tuple(stops) == STOPS_S:
        plot_timecourse(records, summaries, out / "organizer_metric_timecourse.png")
    root = Path(__file__).resolve().parent
    output_files = ["organizer_metric_timecourse.csv", "organizer_metric_summary.csv",
                    "organizer_metric_aggregate.csv"]
    if make_plot and tuple(stops) == STOPS_S:
        output_files.append("organizer_metric_timecourse.png")
    receipt = dict(scope="Exploratory re-use of previously inspected test MAT files",
                   organizer_rule="max_time(mean_trial(binary correctness))",
                   distinction="Peak is test-label-selected; fixed time is descriptive, not newly blind",
                   modeling="Separate full-training CSP+LDA model at each causal one-second endpoint",
                   fixed_time_s=float(fixed_s), endpoints_s=list(stops),
                   window_s=WINDOW_S, model=MODEL, bands_hz=list(map(list, BANDS)),
                   input_sha256=hashes,
                   source_hash_normalization="CRLF to LF for source only; inputs and outputs use raw bytes",
                   source_sha256={name: _source_digest(root / name) for name in
                                  ("organizer_metric.py", "comparison.py", "models.py",
                                   "riemann.py", "features.py", "data.py")},
                   output_sha256={name: _digest(out / name) for name in output_files},
                   historical_timecourse_parity=reference_csv is not None,
                   runtime=dict(python=platform.python_version(), numpy=np.__version__,
                                scipy=scipy.__version__, sklearn=sklearn.__version__,
                                matplotlib=__import__("matplotlib").__version__ if make_plot else None),
                   limitations="Peak is not prospective; stimulation times and device latency unavailable")
    (out / "organizer_metric_provenance.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return records, summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/stroke-rehab"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/organizer_metric"))
    parser.add_argument("--reference-csv", type=Path, default=None,
                        help="Fail if the older CSP timecourse disagrees")
    args = parser.parse_args(argv)
    rows, summary = run(args.data_dir, args.output_dir, reference_csv=args.reference_csv)
    print(f"Saved {len(rows)} time points and {len(summary)} session peak/fixed summaries")


if __name__ == "__main__":
    main()
