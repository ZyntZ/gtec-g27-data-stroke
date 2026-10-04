"""An output-free main must not silently skip physiology parity checks."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("artifact_verification", Path(__file__).parents[1] / "analysis/overnight/verify_artifacts.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)
NAMES = ("all_training_trial_mu.csv", "p1_pre_training_trial_mu.csv", "training_mu_summary.csv")


@pytest.fixture
def reference(tmp_path):
    (tmp_path / "mu_power").mkdir()
    refs = []
    for name in NAMES:
        content = b"value\r\n0.25\r\n"
        (tmp_path / "mu_power" / name).write_bytes(content)
        refs.append(dict(file="mu_power/" + name, reference_sha256=hashlib.sha256(content).hexdigest()))
    return tmp_path, dict(source_commit="frozen", csv_checks=refs)


def test_all_three_frozen_csvs_are_checked(reference):
    out, manifest = reference
    assert len(VERIFY.verify_physiology_csvs(out, manifest)) == 3


@pytest.mark.parametrize("mode", ["missing", "changed", "empty_manifest"])
def test_missing_or_changed_reference_cannot_report_a_pass(reference, mode):
    out, manifest = reference
    if mode == "missing":
        (out / "mu_power" / NAMES[0]).unlink()
    elif mode == "changed":
        (out / "mu_power" / NAMES[0]).write_bytes(b"value\r\n0.26\r\n")
    else:
        manifest["csv_checks"] = []
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        VERIFY.verify_physiology_csvs(out, manifest)
