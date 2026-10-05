"""Build the one-page review from the integration verification receipt.

Requires reportlab. Run from the repository root after verification.
"""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]


def build():
    receipt = json.loads((ROOT / "results/pr_reliability/verification.json").read_text())
    assert receipt["early_all_output_bytes_unchanged"]
    assert receipt["early_recording_hashes_unchanged"]
    assert receipt["tests"]["returncode"] == receipt["preflight"]["returncode"] == 0
    target = ROOT / "output/pdf/g27_pr_reliability_review.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    body = ParagraphStyle("Body", fontName="Helvetica", fontSize=10,
                          leading=13.3, textColor=colors.HexColor("#18232b"), spaceAfter=8)
    title = ParagraphStyle("Title", parent=body, fontName="Helvetica-Bold",
                           fontSize=19, leading=23, textColor=colors.black, spaceAfter=7)
    heading = ParagraphStyle("Heading", parent=body, fontName="Helvetica-Bold",
                             fontSize=11, leading=15, spaceBefore=6, spaceAfter=5)
    small = ParagraphStyle("Small", parent=body, fontSize=8.5, leading=11, spaceAfter=6)
    cell = ParagraphStyle("Cell", parent=body, fontSize=9.4, leading=12.4,
                          spaceAfter=0, alignment=TA_LEFT)
    p = lambda s, style=body: Paragraph(s, style)
    content = [p("G27 Pull Request Reliability Review", title),
               p("5 October 2026 | Findings and verified integration changes", small),
               p("We reproduced the combined-branch failure and repaired the source receipt "
                 "by rerunning the frozen analysis. Normal prediction parity had missed "
                 "error paths; passing branches separately had missed their integration. "
                 "The numerical findings remain unchanged.")]
    rows = [[p("Finding", heading), p("Cause and response", heading)],
            [p("PR5 model state", cell), p("State advanced before the classifier returned. "
             "Included PR8 commits state only after success and restores it after a failed batch.", cell)],
            [p("PR8 EEG stream", cell), p("Model rollback did not restore consumed EEG. Updated "
             "PR8 fails closed: a processing error makes the decoder terminal. Recovery "
             "reconstructs it and replays from a known run boundary; chunk retry is unsupported.", cell)],
            [p("PR7/8 source receipts", cell), p("PR7 rejected the changed Riemannian source; PR8's "
             "test ignored extra stream-source hashes. Actual reruns preserve historical evidence. "
             "Tests now verify all declared current sources, output hashes and prediction correctness.", cell)],
            [p("PR6 evidence labels", cell), p("Random balanced subsets were easy to mistake for "
             "chronological calibration. PNG, SVG and HTML captions now state the limitation. "
             "Historical receipts and convenience exports have an explicit scope.", cell)]]
    table = Table(rows, colWidths=[102, 397], hAlign="LEFT")
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f4")),
                               ("LEFTPADDING", (0, 0), (-1, -1), 7),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                               ("TOPPADDING", (0, 0), (-1, -1), 7),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    content.extend([table, Spacer(1, 9), p("Verification and prevention", heading),
        p(f"{receipt['tests']['summary']}. Preflight and published-result arithmetic pass. "
          "All 24 early choices, 1,920 outer and 1,080 inner predictions match; "
          "3,840 historical forward predictions and 192 summaries reproduce. "
          "Automated checks run on pull requests, pushes and merge groups without raw EEG. "
          + (f"GitHub: {receipt['hosted_checks']['summary']} on "
             f"{receipt['hosted_checks']['head_sha'][:7]} (Python 3.11, optional accuracy dependencies installed)."
             if 'hosted_checks' in receipt else
             "Hosted workflow execution is reported separately from these local checks.")),
        p("Scientific result and limits", heading),
        p("At 60 calibration labels and +3.5 seconds, fixed CSP scores <b>82/120</b> "
          "versus <b>75/120</b> for selection; P2 PRE falls from 13/20 to 8/20. "
          "These exposed retrospective tails are not independent validation. "
          "Feedback onset and clinical side remain uncertain. Repairing reproducibility "
          "does not establish motor-intent specificity or rehabilitation benefit."),
        p("Sources: Anna's comments on "
          '<a href="https://github.com/ZyntZ/gtec-g27-data-stroke/pull/5#issuecomment-5989971381">PR5</a>, '
          '<a href="https://github.com/ZyntZ/gtec-g27-data-stroke/pull/8#issuecomment-5989985404">PR8</a>, '
          '<a href="https://github.com/ZyntZ/gtec-g27-data-stroke/pull/7#issuecomment-5989993987">PR7</a> and '
          '<a href="https://github.com/ZyntZ/gtec-g27-data-stroke/pull/6#issuecomment-5990009998">PR6</a>. '
          "Requested independent reviewer: Astra / Ultra; runtime identity unverified. "
          "Exact hashes: results/pr_reliability/verification.json. "
          "Candidate integration only; no main merge or competition submission.", small)])
    SimpleDocTemplate(str(target), pagesize=A4, leftMargin=48, rightMargin=48,
                      topMargin=40, bottomMargin=40, title="G27 Pull Request Reliability Review",
                      author="").build(content)
    print(target)


if __name__ == "__main__":
    build()
