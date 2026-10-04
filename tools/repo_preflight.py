"""Reject private data, generated caches and failed notebooks from a Git commit.

Usage: python tools/repo_preflight.py

The check uses Git's *staged/tracked* file list rather than scanning the local
organizer archive, which should remain in ignored data/. No data are uploaded.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import subprocess
import sys


FORBIDDEN_SUFFIXES = {".mat", ".rar", ".pyc", ".pyo", ".joblib", ".pkl", ".partial"}
FORBIDDEN_NAMES = {".env", "Thumbs.db", ".DS_Store"}
FORBIDDEN_DIRS = {"data", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".venv"}


def violations(paths: list[str]) -> list[str]:
    """Return source-path policy violations; accepts paths from `git ls-files`."""
    errors = []
    for raw in paths:
        name = PurePosixPath(raw)
        if (name.is_absolute() or ".." in name.parts or "\\" in raw
                or any(part in FORBIDDEN_DIRS for part in name.parts)
                or name.name in FORBIDDEN_NAMES
                or name.suffix.lower() in FORBIDDEN_SUFFIXES):
            errors.append(raw)
    return errors


def notebook_errors(path: Path) -> list[str]:
    """Check committed notebook JSON, error outputs and local-path disclosures."""
    problems = []
    try:
        book = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"{path}: cannot read notebook JSON: {exc}"]
    if book.get("nbformat") != 4 or not isinstance(book.get("cells"), list):
        return [f"{path}: expected notebook format 4 with a cell list"]
    for n, cell in enumerate(book["cells"]):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                problems.append(f"{path}: cell {n + 1} contains an error output")
            # Generated outputs must not expose a developer's local paths.
            rendered = json.dumps(output, ensure_ascii=False)
            if any(prefix in rendered for prefix in ("/workspace/", "/Users/", "/home/")):
                problems.append(f"{path}: cell {n + 1} output exposes a local path")
    return problems


def check_repository(root: Path) -> list[str]:
    """Check precisely the staged/tracked files that Git would upload."""
    result = subprocess.run(["git", "ls-files", "--cached", "-z"], cwd=root,
                            capture_output=True, check=False)
    if result.returncode:
        return ["git ls-files failed; run this script from a Git checkout"]
    paths = [value.decode("utf-8", "surrogateescape") for value in
             result.stdout.split(b"\0") if value]
    errors = [f"Do not track {name}" for name in violations(paths)]
    for name in paths:
        if name.endswith(".ipynb"):
            errors.extend(notebook_errors(root / name))
    return errors


def main() -> int:
    errors = check_repository(Path(__file__).resolve().parent.parent)
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print("Repository preflight passed: no tracked raw data, caches or failed notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
