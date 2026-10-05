"""Can the recordings show when feedback starts? Two label-free probes per file.

The files hold no stimulation marker. The paper states the stimulator runs at
50 Hz, so with the 50 Hz notch already applied its trace could only survive at
harmonics: 100 Hz, and 150/200 Hz folded to 106/56 Hz at 256 Hz sampling.
Probe 1 asks whether power at those lines changes between the pre-instruction
rest and the nominal feedback phase. Probe 2 asks whether the trial-averaged
EEG shows a response locked to the nominal feedback start, as it does to the
instruction. A null result does not mean stimulation was absent or late.
"""
from pathlib import Path
import csv

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .data import read_recording, session_paths

STIMULATION_LINES_HZ = {"line_100hz": 100.0, "line_106hz_alias_of_150": 106.0,
                        "line_56hz_alias_of_200": 56.0}
REST_S = (0.5, 2.0)        # after the attention sound, before the instruction
FEEDBACK_S = (4.0, 8.0)    # inside the documented feedback phase
EVENTS_S = {"instruction_2p0s": 2.0, "nominal_feedback_3p5s": 3.5, "relax_8p0s": 8.0}
EVOKED_SPAN_S = 0.75


def _bandpass(recording, low, high):
    sos = butter(4, (low, high), btype="bandpass", fs=recording.fs, output="sos")
    return sosfiltfilt(sos, recording.signal, axis=0)


def line_change_db(recording, hz, *, half_width=1.0):
    """Median over trials and channels: feedback-phase power relative to rest (dB)."""
    power = _bandpass(recording, hz - half_width, hz + half_width) ** 2
    fs = recording.fs
    mean = lambda onset, span: power[onset + round(span[0] * fs):
                                     onset + round(span[1] * fs)].mean(axis=0)
    ratio = [mean(o, FEEDBACK_S) / mean(o, REST_S) for o in recording.onsets]
    return float(np.median(10 * np.log10(ratio)))


def evoked_ratios(recording):
    """Size of the trial-median 1-12 Hz response after each event, relative to rest.

    Values near 1 mean nothing is time-locked to that event. Trials too close
    to the end of the file for the last window are left out of that window.
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
            row |= {f"{name}_feedback_vs_rest_db": round(line_change_db(recording, hz), 2)
                    for name, hz in STIMULATION_LINES_HZ.items()}
            row |= {f"evoked_{name}_vs_rest": round(value, 2)
                    for name, value in evoked_ratios(recording).items()}
            rows.append(row)
            print(f"{patient} {stage} {run}: " + ", ".join(
                f"{k.split('_feedback')[0].split('_vs_rest')[0]} {v}"
                for k, v in list(row.items())[3:]), flush=True)
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
