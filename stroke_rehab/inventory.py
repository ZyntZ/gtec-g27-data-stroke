"""Build a compact, reproducible data-quality inventory (no raw EEG)."""
import csv
from pathlib import Path
import hashlib

import numpy as np

from .data import read_recording, session_paths


def build_inventory(data_root, output_file):
    records = []
    for patient, stage, training_path, test_path in session_paths(data_root):
        for role, path in (("training", training_path), ("test", test_path)):
            rec = read_recording(path)
            with Path(path).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            records.append({"file": Path(path).name, "patient": patient,
                            "session": stage, "run": role, "fs_hz": rec.fs,
                            "samples": len(rec.signal), "channels": rec.signal.shape[1],
                            "trials": len(rec.labels),
                            "left_trials": int(np.sum(rec.labels == 1)),
                            "right_trials": int(np.sum(rec.labels == -1)),
                            "sha256": digest})
    destination = Path(output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records
