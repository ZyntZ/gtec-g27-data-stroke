"""Data-free guards for the navigable methods guide and quoted counts."""
from __future__ import annotations

import csv
from pathlib import Path
import re
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _rows(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda path: str(path.relative_to(ROOT)))
def test_documentation_links_point_to_existing_files(document):
    """A reader must not encounter broken relative links or untracked figures."""
    text = document.read_text(encoding="utf-8")
    for link in LINK.findall(text):
        target, _, fragment = link.partition("#")
        target = unquote(target)
        if re.match(r"^(https?://|mailto:)", target):
            continue
        resolved = (document.parent / target).resolve() if target else document
        assert resolved.is_relative_to(ROOT.resolve()), (document, link)
        assert resolved.exists(), (document, link)
        if fragment and resolved.suffix == ".md":
            # Heading slugs used for in-repo section links are explicit and simple.
            headings = [line.lstrip("# ").strip().lower().replace(" ", "-")
                        for line in resolved.read_text(encoding="utf-8").splitlines()
                        if line.startswith("#")]
            assert fragment in headings, (document, link)


def test_public_headline_counts_match_saved_data():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs/experiments.md").read_text(encoding="utf-8")
    index = (ROOT / "docs/index.md").read_text(encoding="utf-8")
    forward = _rows("results/forward_calibration/forward_calibration_summary.csv")
    counts = {}
    for t in (2.0, 3.5, 4.25, 6.5):
        subset = [row for row in forward if row["model"] == "filterbank_csp_shrinkage_lda"
                  and int(row["calibration_trials"]) == 60
                  and float(row["decision_time_s"]) == t]
        assert len(subset) == 6 and {row["trials"] for row in subset} == {"20"}
        counts[t] = sum(int(row["correct"]) for row in subset)
    assert counts == {2.0: 55, 3.5: 82, 4.25: 99, 6.5: 99}
    for count in counts.values():
        assert f"{count}/120" in guide
    assert "82/120" in readme and "99/120" in readme
    assert "82/120" in index and "99/120" in index

    selected = _rows("results/early_selection/early_summary.csv")
    n = sum(int(row["correct"]) for row in selected if
            row["strategy"] == "training_selected" and int(row["calibration_trials"]) == 60)
    assert n == 75
    assert "75/120" in readme and "75/120" in guide and "75/120" in index

    organizer = _rows("results/organizer_metric/organizer_metric_aggregate.csv")
    assert len(organizer) == 1
    row = organizer[0]
    for field, expected in (("fixed_correct", 358),
                            ("pooled_peak_correct", 407),
                            ("sum_of_session_peaks_correct", 427)):
        assert int(row[field]) == expected
        assert f"{expected}/480" in index


def test_landing_page_is_short_and_explains_nonblind_test():
    home = (ROOT / "README.md").read_text(encoding="utf-8")
    assert 30 <= len(home.splitlines()) <= 125
    assert "not blind validation" in home
    assert "no per-trial stimulation timestamps" in home
    for name in ("index.md", "protocol.md", "experiments.md"):
        assert (ROOT / "docs" / name).is_file()
