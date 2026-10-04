"""Cross-session transfer: decode a session with no calibration data from that day.

The model is fitted on the training run of the SAME patient's other session and
applied to the target run; no test run is ever used for fitting. Every run is recentered by its own covariance average,
without labels; the target run is recentered causally, starting from its first
trial. Channels are matched by NAME through `montage.layout_for`, because the
two P1 sessions appear to use different electrode layouts. `columns_as_stored`
skips that matching for comparison.
"""
from pathlib import Path
import csv

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import riemann
from .data import read_recording, session_paths
from .montage import LAYOUTS, layout_for

WINDOW = (2.5, 6.5)
COMMON = tuple(name for name in LAYOUTS["paper"] if name in LAYOUTS["montage_png"])


def _covariances(path, names):
    recording = read_recording(path)
    if names is not None:
        layout = layout_for(path)
        recording = type(recording)(recording.signal[:, [layout.index(n) for n in names]],
                                    recording.labels, recording.onsets, recording.fs)
    trials = riemann.filtered_trials(recording)
    return riemann.window_covariances(trials, recording.fs, WINDOW), recording.labels


def _features(covariances, *, online):
    """Tangent vectors after recentering a run on itself (batch) or causally (online)."""
    rows = []
    means = None if online else [riemann.riemann_mean(covariances[:, b])
                                 for b in range(covariances.shape[1])]
    for i, trial in enumerate(covariances):
        if online:
            means = (list(trial) if i == 0 else
                     [riemann.geodesic(m, c, 1.0 / (i + 1)) for m, c in zip(means, trial)])
        rows.append(np.concatenate([riemann.tangent_vector(c, m)
                                    for c, m in zip(trial, means)]))
    return np.array(rows)


def run_transfer(data_root, output_file):
    paths = {(p, s): (train, test) for p, s, train, test in session_paths(data_root)}
    rows = []
    for channels, names in (("common_names", COMMON), ("columns_as_stored", None)):
        cache = {path: _covariances(path, names) for pair in paths.values() for path in pair}
        for (patient, stage), targets in paths.items():
            source = paths[patient, "post" if stage == "pre" else "pre"][0]
            x, y = _features(cache[source][0], online=False), cache[source][1]
            model = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=0.01, max_iter=3000)).fit(x, y)
            for run, path in zip(("training", "test"), targets):
                covariances, labels = cache[path]
                predicted = model.predict(_features(covariances, online=True))
                rows.append(dict(patient=patient, session=stage, target_run=run,
                                 source_session="post" if stage == "pre" else "pre",
                                 source_run="training",
                                 channels=channels,
                                 n_channels=covariances.shape[-1],
                                 correct=int((predicted == labels).sum()),
                                 trials=len(labels)))
                print(f"{patient} {stage} {run} from other session, {channels}: "
                      f"{rows[-1]['correct']}/80", flush=True)
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
