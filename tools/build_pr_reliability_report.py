"""Build the one-page report for its explicitly recorded code snapshot.

Requires reportlab. Does not rerun EEG analyses or certify newer source trees.
"""
import csv
import hashlib
import json
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]


def report_evidence():
    receipt = json.loads((ROOT / "results/pr_reliability/verification.json").read_text())
    snapshot = json.loads((ROOT / "results/pr_reliability/report_snapshot.json").read_text())
    current = receipt["current_stack_checks"]
    for section in ("source_sha256_lf", "checks_source_sha256_lf"):
        for name, expected in current[section].items():
            actual = hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            if actual != expected:
                raise ValueError(f"Report snapshot no longer describes {name}; reverify before rebuilding")
    for name, expected in current["organizer_metric"]["output_sha256"].items():
        actual = hashlib.sha256((ROOT / "results/organizer_metric" / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Report snapshot no longer describes {name}")
    commands = current["final_integration_checks"]["commands"]
    if any(command["returncode"] != 0 for command in commands):
        raise ValueError("Recorded combined verification did not pass")
    test_output = next(command["output"] for command in commands if command["command"] == "python -m pytest -q")
    match = re.search(r"(\d+) passed", test_output)
    if not match or int(match.group(1)) != snapshot["hosted_checks"]["tests_passed"]:
        raise ValueError("Local and hosted snapshot test counts disagree")
    if snapshot["hosted_checks"]["head_sha"] != snapshot["reviewed_candidate_commit"]:
        raise ValueError("Hosted check refers to a different report snapshot")
    if snapshot["hosted_checks"]["conclusion"] != "success":
        raise ValueError("Hosted snapshot verification did not succeed")
    with (ROOT / "results/organizer_metric/organizer_metric_aggregate.csv").open(newline="") as source:
        aggregate = next(csv.DictReader(source))
    return receipt, snapshot, int(match.group(1)), aggregate


def build():
    receipt, snapshot, tests_passed, aggregate = report_evidence()
    target = ROOT / "output/pdf/g27_pr_reliability_review.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    body = ParagraphStyle("Body", fontName="Helvetica", fontSize=9.5,
                          leading=12, textColor=colors.HexColor("#18232b"), spaceAfter=6)
    title = ParagraphStyle("Title", parent=body, fontName="Helvetica-Bold",
                           fontSize=18, leading=22, textColor=colors.black, spaceAfter=5)
    heading = ParagraphStyle("Heading", parent=body, fontName="Helvetica-Bold",
                             fontSize=10.5, leading=13, spaceBefore=5, spaceAfter=4)
    small = ParagraphStyle("Small", parent=body, fontSize=8, leading=10, spaceAfter=5)
    cell = ParagraphStyle("Cell", parent=body, fontSize=9, leading=11.5,
                          spaceAfter=0, alignment=TA_LEFT)
    p = lambda s, style=body: Paragraph(s, style)
    code = snapshot["reviewed_candidate_commit"][:7]
    content = [p("G27 Pull Request Reliability Review", title),
               p(f"Version {snapshot['report_version']} | 5 October 2026 | Code snapshot {code}", small),
               p("The combined implementation addresses Anna's reported state and source-receipt "
                 "failures, including main's update03. Recorded reruns preserve the numerical "
                 "outputs. This page replaces the earlier 127/130-test report; it certifies "
                 "the named snapshot, not future commits.")]
    rows = [[p("Finding", heading), p("Cause and response", heading)],
            [p("Model state", cell), p("PR5 advanced state before classification succeeded. PR8 "
             "commits state after success and restores model state after a failed batch.", cell)],
            [p("Consumed EEG", cell), p("Model rollback did not rewind the EEG stream. Updated PR8 "
             "makes processing failures terminal; recovery reconstructs the decoder and replays "
             "from a known run boundary. Retrying a consumed chunk is rejected.", cell)],
            [p("Combined receipts", cell), p("PR7/8 integration and then update03 changed sources "
             "behind saved receipts. Actual reruns refresh active evidence and preserve historical "
             "snapshots. Guards check source/output hashes and prediction correctness.", cell)],
            [p("Evidence labels", cell), p("Balanced random subsets are retrospective, not "
             "chronological calibration. Reports label inspected test reuse, historical checks "
             "and the distinct scoring windows.", cell)]]
    table = Table(rows, colWidths=[100, 399], hAlign="LEFT")
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f4")),
                               ("LEFTPADDING", (0, 0), (-1, -1), 7),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                               ("TOPPADDING", (0, 0), (-1, -1), 6),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    total = int(aggregate["test_trials"])
    content.extend([table, Spacer(1, 7), p("Verification and independent review", heading),
        p(f"<b>{tests_passed} tests pass</b> locally and in "
          f'<a href="{snapshot["hosted_checks"]["url"]}">GitHub CI</a>; preflight and 19 published-result '
          "checks pass. Training replay preserves 3,840 predictions, 192 summaries and six "
          "80-trial raw runs. Organizer reruns preserve three CSVs and 12 input hashes."),
        p("The requested Astra/Ultra review is recorded as completed; actual runtime identity "
          "is unverified. A completed read-only review verified <b>Claude Fable 5.1 at Max</b> "
          f"on {code}, with no blocking code or provenance finding. Its medium report-freshness "
          "finding is corrected here. Smaller hardening suggestions are recorded in the companion."),
        p("Scientific results and limits", heading),
        p(f"Organizer-style CSP: fixed +3.5 s <b>{aggregate['fixed_correct']}/{total}</b>; "
          f"one pooled peak <b>{aggregate['pooled_peak_correct']}/{total}</b>; sum of six separately "
          f"selected session peaks <b>{aggregate['sum_of_session_peaks_correct']}/{total}</b>. "
          "Peak times were selected after seeing test results. Offline 433/480 baseline and "
          "exploratory Riemannian 446/480 use a different scoring contract."),
        p("Early training-tail selection remains <b>75/120 versus fixed CSP 82/120</b>. "
          "There is no independent accuracy improvement, motor-intent or rehabilitation claim. "
          "Actual FES onset, clinical side and physical channel geometry remain unconfirmed."),
        p("Integration and delivery", heading),
        p("Anna merged PR6 into main on 5 October 2026. It includes PR5, PR7, PR8 and PR9; "
          "GitHub also marks PR5/7/9 merged. PR8 remains open although its commits are included. "
          "This report update is a separate documentation change. Video/submission remain team outcomes."),
        p("Evidence: results/pr_reliability/verification.json (current_stack_checks), "
          "report_snapshot.json, and docs/pr_reliability_review.md. "
          '<a href="https://github.com/ZyntZ/gtec-g27-data-stroke/pull/6#issuecomment-5992153702">'
          "Anna's update03 request</a>. Workflow applies to PRs containing it, pushes to "
          "main/codex/pr-reliability, and enabled merge groups. CI does not rerun raw EEG.", small)])
    SimpleDocTemplate(str(target), pagesize=A4, leftMargin=48, rightMargin=48,
                      topMargin=36, bottomMargin=36, title="G27 Pull Request Reliability Review - Version 2",
                      author="").build(content)
    print(target)


if __name__ == "__main__":
    build()
