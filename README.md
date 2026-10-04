# Stroke Rehab / G27

A reproducible starting line for the BR41N.IO Stroke Rehab track. We use the organizer's EEG recordings, keep training and test runs separate, and measure what our decoder actually does on **80 held-out trials per session**. Two people, six sessions, no mystery score.

## What is here

- A verified download of the organizer archive (SHA-256 checked before extraction). The recordings stay in ignored `data/`, not Git or the update zip.
- A strict loader for the unusual trigger format: **one label repeated for 2,048 samples**, not 2,048 separate examples.
- A 3-band EEG feature bank (8–12, 12–20, 20–30 Hz), a shrinkage linear discriminant analysis (LDA) baseline, and a regularized filter-bank common spatial patterns (CSP) model.
- Training-run-only 5-fold trial-level cross-validation (CV), followed by one held-out test evaluation for each patient × session.
- A fixed early/late timing audit. This matters: a late high score is not automatically clean evidence of *motor-intent* decoding when the test run can contain conditional feedback.
- Provenance, machine-readable scores, a figure, and synthetic unit tests. See [the protocol](docs/protocol.md) before comparing numbers with anyone else's.

## Get running

Python 3.10+ and a system `libarchive` library are needed for the RAR reader (for example, `libarchive13` on Debian/Ubuntu). From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
stroke-rehab download
stroke-rehab inventory
stroke-rehab run
stroke-rehab audit
stroke-rehab plot
pytest -q
```

`stroke-rehab download` fetches ~175 MB to `data/stroke-rehab.rar` and expands the relevant files to `data/stroke-rehab/`. The downloaded archive and EEG files are intentionally excluded from commits and incremental update zips. If you already have the official archive, put it at `data/stroke-rehab.rar` before running `download`; the SHA-256 check still runs. For a different path, use `--archive` and `--data-dir`. Outputs go to `results/` by default. Rebuilding results requires downloading the dataset.

## First held-out run

Selection is based on training-run CV alone. One final prediction is made for each test trial from **2.5–6.5 seconds after trigger onset**; the instruction is given at **2 seconds**. Numbers below are from the committed `results/session_results.csv`, seed 27.

| Patient | PRE: correct / 80 | PRE accuracy | POST: correct / 80 | POST accuracy |
|:--|--:|--:|--:|--:|
| P1 | 74 | 92.5% | 74 | 92.5% |
| P2 | 66 | 82.5% | 79 | 98.75% |
| P3 | 70 | 87.5% | 70 | 87.5% |

Overall: **433/480 = 90.2%** over these *six observed runs* (not 480 independent people). Both classes have 40 test trials per run. CSP won training CV in all six sessions; the other candidate's training CV scores are in `results/train_cv_candidates.json`. [View the figure](results/session_accuracy.png).

The time audit is a warning, not a victory lap. With fixed CSP, the **2.5–3.5 s** window scored 364/480 (75.8%); the **4.5–6.5 s** window scored 412/480 (85.8%). In particular, P2 PRE scored 52/80 early versus 66/80 late. See `results/timing_audit.csv`. The late rise could reflect actual imagery, feedback-related activity, or both. This dataset does not isolate these explanations. This offline score is neither proof of a rehabilitation benefit nor a real-time deployment result.

The organizer's `overview.pdf` lists another pair of methods and six scores, but it evaluates a different pipeline/time-scoring rule; **those percentages are not a head-to-head leaderboard comparison**. Do not tune future models to the already inspected test labels and then describe the same tests as fresh validation. For what to build next, see [next steps](docs/next_steps.md).

## Layout

```text
stroke_rehab/                 loader, DSP, models, training-only selection, audits
results/data_inventory.csv    sizes, class counts, SHA-256 for each MAT file
results/session_results.csv   per-run selected-model held-out metrics
results/train_cv_candidates.json
results/timing_audit.csv
results/session_accuracy.png
tests/                       synthetic, data-free checks
docs/                        protocol and follow-up experiments
data/                        locally downloaded archive and MAT files (ignored)
```

The source download is [g.tec's Stroke Rehab archive](https://www.gtec.at/downloads_QyTs23/Hackathon/stroke-rehab.rar). See its `DatasetInformation.pdf` and `overview.pdf` for recording details. These recordings and the supplied PDFs are not redistributed here; use the organizer's access terms when sharing the data.
