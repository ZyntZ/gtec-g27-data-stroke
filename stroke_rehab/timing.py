"""Do the recordings show when feedback starts? Two simple probes per file.

The files hold no stimulation marker. The paper states the stimulator runs at
50 Hz. Ten files carry a 50 Hz notch, so there a trace could only survive at
harmonics: 100 Hz, and 150/200 Hz folded to 106/56 Hz at 256 Hz sampling. The
two P1 POST files carry a 60 Hz notch instead, so 50 Hz itself is still
present there, mixed with mains interference at the same frequency.
Probe 1 asks whether power at those lines changes between the pre-instruction
rest and the nominal feedback phase, pooled and per imagined hand. Probe 2
asks whether the trial-median EEG shows a response locked to the nominal
feedback start, as it does to the instruction.

These probes look for a FIXED onset shared by most trials. Probe 2 pools both
hands and takes a median over trials, so it nearly misses a response of
opposite sign for the two hands or one whose latency varies from trial to
trial (see tests/test_timing.py). A null result therefore means "no reliable
onset found by these probes", not that feedback was absent, late or untimed.
"""
from pathlib import Path
import csv

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .data import read_recording, session_paths
from .montage import notch_hz

STIMULATION_LINES_HZ = {"line_50hz": 50.0, "line_100hz": 100.0,
                        "line_106hz_alias_of_150": 106.0, "line_56hz_alias_of_200": 56.0}
HANDS = {"pooled": None, "left": 1, "right": -1}
REST_S = (0.5, 2.0)        # after the attention sound, before the instruction
FEEDBACK_S = (4.0, 8.0)    # inside the documented feedback phase
EVENTS_S = {"instruction_2p0s": 2.0, "nominal_feedback_3p5s": 3.5, "relax_8p0s": 8.0}
EVOKED_SPAN_S = 0.75


def _bandpass(recording, low, high):
    sos = butter(4, (low, high), btype="bandpass", fs=recording.fs, output="sos")
    return sosfiltfilt(sos, recording.signal, axis=0)


def line_change_db(recording, hz, *, half_width=1.0, label=None):
    """Median over trials and channels: feedback-phase power relative to rest (dB).

    `label` restricts the trials to one imagined hand (+1 left, -1 right).
    """
    power = _bandpass(recording, hz - half_width, hz + half_width) ** 2
    fs = recording.fs
    mean = lambda onset, span: power[onset + round(span[0] * fs):
                                     onset + round(span[1] * fs)].mean(axis=0)
    ratio = [mean(o, FEEDBACK_S) / mean(o, REST_S)
             for o, trial_label in zip(recording.onsets, recording.labels)
             if label is None or trial_label == label]
    return float(np.median(10 * np.log10(ratio)))


def evoked_ratios(recording):
    """Size of the trial-median 1-12 Hz response after each event, relative to rest.

    Values near 1 mean no response with a fixed latency and a sign shared by
    both hands. Trials too close to the end of the file for the last window
    are left out of that window.
    """
    filtered = _bandpass(recording, 1.0, 12.0)
    fs = recording.fs

    def size(start, stop):
        lo, hi = round(start * fs), round(stop * fs)
        epochs = [filtered[o + lo:o + hi] - filtered[o + round(REST_S[0] * fs):
                                                     o + round(REST_S[1] * fs)].mean(axis=0)
                  for o in recording.onsets if o + hi <= len(filtered)]
        return float(np.sqrt(np.mean(np.median(epochs, axis=0) ** 2)))

    rest = size(REST_S[1] - EVOKED_SPAN_S, REST_S[1])
    return {name: size(t, t + EVOKED_SPAN_S) / rest for name, t in EVENTS_S.items()}


def run_feedback_check(data_root, output_file):
    rows = []
    for patient, stage, train_path, test_path in session_paths(data_root):
        for run, path in (("training", train_path), ("test", test_path)):
            recording = read_recording(path)
            row = dict(patient=patient, session=stage, run=run)
            row["notch_hz"] = notch_hz(recording)
            row |= {f"{name}_{hand}_feedback_vs_rest_db":
                    round(line_change_db(recording, hz, label=label), 2)
                    for name, hz in STIMULATION_LINES_HZ.items()
                    for hand, label in HANDS.items()}
            row |= {f"evoked_{name}_vs_rest": round(value, 2)
                    for name, value in evoked_ratios(recording).items()}
            rows.append(row)
            print(f"{patient} {stage} {run}: " + ", ".join(
                f"{k.split('_feedback')[0].split('_vs_rest')[0]} {v}"
                for k, v in list(row.items())[4:] if "_left_" not in k and "_right_" not in k),
                flush=True)
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
