"""Repository guard: reject training/test data and leaked notebook state."""
import importlib.util
import json
from pathlib import Path
import subprocess


SOURCE = Path(__file__).resolve().parents[1] / "tools" / "repo_preflight.py"
SPEC = importlib.util.spec_from_file_location("repo_preflight", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_path_policy():
    safe = ["README.md", "stroke_rehab/data.py", "results/causal_nested_summary.csv",
            "notebooks/01_training_eda.ipynb", ".github/workflows/ci.yml"]
    assert MODULE.violations(safe) == []
    forbidden = ["data/P1_pre_test.mat", "stroke-rehab.rar", "models/P1.joblib",
                 "src/__pycache__/f.pyc", ".env", "data/stroke-rehab/overview.pdf",
                 "notebooks/.ipynb_checkpoints/0.ipynb"]
    assert MODULE.violations(forbidden) == forbidden


def test_notebook_guard(tmp_path):
    path = tmp_path / "checked.ipynb"
    path.write_text(json.dumps({"nbformat": 4, "cells": [{"cell_type": "code",
                      "outputs": [{"output_type": "stream", "text": "ok"}]}]}))
    assert not MODULE.notebook_errors(path)
    path.write_text(json.dumps({"nbformat": 4, "cells": [{"cell_type": "code",
                      "outputs": [{"output_type": "error", "evalue": "oops"},
                                  {"output_type": "stream", "text": "/workspace/private"}]}]}))
    assert len(MODULE.notebook_errors(path)) == 2
    path.write_text("not valid json")
    assert len(MODULE.notebook_errors(path)) == 1


def test_git_index_is_checked_instead_of_local_data(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "README.md").write_text("test")
    (tmp_path / "data").mkdir()
    (tmp_path / "data/P1_pre_training.mat").write_bytes(b"local-only")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    assert MODULE.check_repository(tmp_path) == []
    subprocess.run(["git", "add", "data/P1_pre_training.mat"], cwd=tmp_path, check=True)
    assert any("P1_pre_training.mat" in error for error in MODULE.check_repository(tmp_path))
