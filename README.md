# Stroke Rehab / G27

A reproducible starting line for the BR41N.IO Stroke Rehab track. We use the organizer's EEG recordings, keep training and test runs separate, and measure what our decoder actually does on **80 held-out trials per session**. Three active collaborators, six sessions, no mystery score.

## Active team

- [Anna Sokolova](https://github.com/ZyntZ)
- [Jason Wang](https://github.com/wjason00)
- [Sudip Sharma](https://github.com/nxxis)

## What is here

- A verified download of the organizer archive (SHA-256 checked before extraction). The recordings stay in ignored `data/`, not Git or the update zip.
- A strict loader for the unusual trigger format: **one label repeated for 2,048 samples**, not 2,048 separate examples.
- A 3-band EEG feature bank (8–12, 12–20, 20–30 Hz), a shrinkage linear discriminant analysis (LDA) baseline, and a regularized filter-bank common spatial patterns (CSP) model.
- Training-run-only 5-fold trial-level cross-validation (CV), followed by one held-out test evaluation for each patient × session.
- A fixed early/late timing audit. This matters: a late high score is not automatically clean evidence of *motor-intent* decoding when the test run can contain conditional feedback.
- Provenance, machine-readable scores, a figure, and synthetic unit tests. See [the protocol](docs/protocol.md) before comparing numbers with anyone else's.

## Get running

Python 3.11+ and a system `libarchive` library are needed for the RAR reader (for example, `libarchive13` on Debian/Ubuntu). From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
stroke-rehab download
stroke-rehab inventory
stroke-rehab diagnose
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

The organizer's `overview.pdf` lists another pair of methods and six scores, but it evaluates a different pipeline/time-scoring rule; **those percentages are not a head-to-head leaderboard comparison**. Do not tune future models to the already inspected test labels and then describe the same tests as fresh validation.

## Layout

```text
stroke_rehab/                 loader, DSP, models, training-only selection, audits
results/data_inventory.csv    sizes, class counts, SHA-256 for each MAT file
results/session_results.csv   per-run selected-model held-out metrics
results/train_cv_candidates.json
results/timing_audit.csv
results/session_accuracy.png
tests/                       synthetic, data-free checks
docs/                        evaluation protocol
data/                        locally downloaded archive and MAT files (ignored)
```

The source download is [g.tec's Stroke Rehab archive](https://www.gtec.at/downloads_QyTs23/Hackathon/stroke-rehab.rar). See its `DatasetInformation.pdf` and `overview.pdf` for recording details. These recordings and the supplied PDFs are not redistributed here; use the organizer's access terms when sharing the data.

## Training-only stress tests and EDA

Run `stroke-rehab diagnose` after downloading, then open
`notebooks/01_training_eda.ipynb` from the repository root (run all cells).
`configs/diagnostics.json` is read by the command; use `--config` to change its
location or `--data-dir` if the archive was extracted elsewhere. The
notebook reads only training-run EEG and saves `results/train_signal_qc.csv`
and `results/train_diagnostics.png`. Reproducing a local analysis with data
outside `data/stroke-rehab` requires setting `STROKE_REHAB_DATA` to the
extracted directory. No MAT recordings belong in a commit.

The controls compare fixed post-cue CSP with a **causally filtered,
pre-instruction** power probe; both are scored on shuffled and contiguous
(purged-neighbour) training-run folds. These are diagnostic out-of-fold
training scores, **not a new test score**, an online classifier, or evidence
that the feedback effect is solved. See [the evaluation contract](docs/protocol.md).

## Causal training-only replay

`stroke-rehab causal-train` reads `configs/causal_training.json`, replays only
six training MAT files in 64-sample chunks, and writes
`results/causal_train_cv.csv`. With data elsewhere:

```bash
stroke-rehab causal-train --data-dir /path/to/extracted/stroke-rehab
pytest -q
```

Each cue onset is passed to the stream only when its chunk arrives. The
stateful filter bank yields a 48-dimensional log-variance feature from the
2.5–3.5 s post-trigger interval (instruction at 2 s). A standardized,
shrinkage linear discriminant analysis (LDA) model is fitted anew inside
each five-fold **chronological, one-neighbour-purged** training-run split.
A separately fitted complete training-run decoder can produce causal
`DecisionEvent` objects from incoming EEG via `CausalTrialDecoder.process`.
The earliest feature interval ends at 3.5 s; 64-sample chunking can add up
to 0.25 s of acquisition delay. The reported per-chunk processing times are
machine-dependent replay measurements, **not** validated device latency.

| Training run | Causal, blocked out-of-fold correct / 80 |
|:--|--:|
| P1 PRE | 59 |
| P1 POST | 70 |
| P2 PRE | 35 |
| P2 POST | 37 |
| P3 PRE | 48 |
| P3 POST | 36 |

These are **training-only** scores, not held-out test results. The existing
offline 2.5–6.5 s scores cannot be compared directly: they use more EEG,
a different model and noncausal filtering. Feedback may occur in the early
window, and this dataset does not provide its timestamps. No FES or clinical
deployment is implied by the replay.

## Causal spatial decoding: nested training-run study

`stroke-rehab nested-train` runs the fixed `configs/spatial_nested.json` design:
causal 8–12/12–20/20–30 Hz Butterworth filtering, a single 2.5–3.5 s
post-trigger window (the cue occurs at 2 s), and streaming, mean-centered
16×16 channel covariance per band. Unlike epoch buffering, inference only
keeps the filter states plus O(3 × 16²) trial statistics. A fold-local,
ridge-regularized common spatial patterns (CSP) estimator transforms each
covariance to log relative spatial variance; standardized shrinkage linear
discriminant analysis (LDA) makes the binary decision. Three candidates are
predefined: 48-dimensional channel power, 1 spatial-filter pair per band,
or 2 spatial-filter pairs per band. The training-only *inner* 3-fold blocked
cross-validation (CV) chooses among them; *outer* 5-fold blocked CV estimates
the resulting selection rule. Each fit excludes one adjacent trial on either
side of its validation block. Both CSP and standardization are fit inside
the folds; the organizer test runs are never loaded. See `docs/protocol.md`.

```bash
stroke-rehab nested-train
stroke-rehab calibrate-spatial
pytest -q
```

For a different extracted archive, use `--data-dir /path/to/stroke-rehab` on
both commands. Nested evaluation writes only the small derived
`results/causal_nested_{summary,folds}.csv`. Calibration saves six fitted
pipelines and SHA-256 provenance under ignored `data/models/` **locally**:
never put `.joblib` models or organizer data into the update ZIP. To use one
trained decoder on a fresh EEG chunk stream:

```python
import json
import joblib
from stroke_rehab.spatial import CausalSpatialDecoder

metadata = json.load(open("data/models/P1_pre.json", encoding="utf-8"))
model = joblib.load("data/models/P1_pre.joblib")  # Trust only files your own run wrote.
decoder = CausalSpatialDecoder(model, candidate=metadata["candidate"],
                               fs=metadata["fs_hz"],
                               channels=metadata["n_channels"],
                               window=tuple(metadata["window_seconds"]))
# Supply EEG chunks in sample order and each *true* trigger sample index on arrival.
# event = decoder.process(chunk, onsets=[trigger_sample] if cue_in_chunk else [])
# A non-None event has onset_sample, decision_sample and predicted label (+1/-1).
```

| Training session | Nested selection | Fixed power | Fixed CSP 1 pair | Fixed CSP 2 pairs |
|:--|--:|--:|--:|--:|
| P1 PRE | 66/80 | 59/80 | 70/80 | 66/80 |
| P1 POST | 77/80 | 70/80 | 76/80 | 78/80 |
| P2 PRE | 40/80 | 35/80 | 46/80 | 43/80 |
| P2 POST | 48/80 | 37/80 | 46/80 | 49/80 |
| P3 PRE | 55/80 | 48/80 | 56/80 | 53/80 |
| P3 POST | 43/80 | 36/80 | 46/80 | 47/80 |

The pooled counts over the six *observed* runs are **329/480 nested**, versus
**285/480 fixed power**, **340/480 fixed CSP 1 pair**, and **336/480 fixed CSP
2 pairs**. This does not demonstrate that nested selection beats the fixed
CSP choices; all these numbers are exploratory training-run scores, not six
independent patients. No new test-set score is claimed. Decisions depend on
2.5–3.5 s EEG and appear by 3.5–3.75 s after the trigger with 64-sample
chunks. Unknown feedback/electrical-stimulation timing still prevents a
feedback-free or medical-device performance claim.

### Latency-versus-feedback sensitivity (not model selection)

`stroke-rehab latency-audit` recomputes a **fixed CSP-1** model on the same
purged training-only folds for four predeclared window endpoints, keeping the
start at 2.5 s. It saves `results/causal_latency_audit.csv`. Run
`notebooks/02_causal_latency.ipynb` from the repository root to regenerate
[the 320-DPI figure](results/causal_latency.png). The six training-run
correct counts (out of 80) are:

| Session | 3.5 s | 4.5 s | 5.5 s | 6.5 s |
|:--|--:|--:|--:|--:|
| P1 PRE | 70 | 73 | 75 | 73 |
| P1 POST | 76 | 77 | 80 | 77 |
| P2 PRE | 46 | 52 | 60 | 71 |
| P2 POST | 46 | 58 | 68 | 76 |
| P3 PRE | 56 | 68 | 70 | 70 |
| P3 POST | 46 | 56 | 65 | 69 |

Pooled descriptive counts are 340, 384, 418 and 436 out of 480, respectively.
**Later is not automatically better**: training feedback may already be
present, and test-run stimulation is contingent on earlier decoding. There
are no precise feedback timestamps in this dataset. Do not use this audit to
select 6.5 s and then report its score as an independent validation result.

## Training-only negative controls

`stroke-rehab controls` reads `configs/negative_controls.json` and only the six
`*_training.mat` recordings. It compares fixed one-pair causal CSP with the
same fold-local shrinkage LDA on (i) an EEG window entirely **before** the
instruction (0.5–1.75 s), and (ii) the early post-cue 2.5–3.5 s window.
Five contiguous validation folds purge a neighboring trial from fitting.
Within each original 16-trial validation block, 99 randomizations shuffle
trial labels while preserving that block's class counts; the *full pipeline*
is refitted in every shuffle. This is a diagnostic, not a new model-selection
endpoint. Results are saved to `results/training_negative_controls.csv`.

```bash
stroke-rehab controls
# Or: stroke-rehab controls --data-dir /path/to/extracted/stroke-rehab
```

| Training run | Before instruction | Early post-cue | Early upper-tail permutation p |
|:--|--:|--:|--:|
| P1 PRE | 38/80 | 70/80 | 0.01 |
| P1 POST | 39/80 | 76/80 | 0.01 |
| P2 PRE | 37/80 | 46/80 | 0.14 |
| P2 POST | 29/80 | 46/80 | 0.12 |
| P3 PRE | 33/80 | 56/80 | 0.01 |
| P3 POST | 39/80 | 46/80 | 0.17 |

The one-sided p values are limited to 0.01 resolution by 99 randomizations;
all sessions and both directions of a possible pre-instruction effect are
reported in the CSV. These exploratory null distributions assume trial-label
exchangeability *within* each temporal block and are not clinical evidence.
In particular, 29/80 in the P2 POST pre-instruction control is below chance;
its two-sided exploratory permutation p is 0.06 and should not be described
as proof of no pre-cue information. EEG before the instruction could reflect
anticipation, residual filter state, or other confounds. Likewise, the early
window is not guaranteed to precede visual or electrical feedback because
per-trial feedback timestamps are unavailable. No new organizer test files or
labels are read by this command.

## Pull-request sanity checks

Run `pytest -q` and `python tools/repo_preflight.py` locally before opening a
pull request. There is no hosted CI workflow in this repository; these checks
run only when someone runs them. Preflight rejects staged MAT/RAR files, serialized estimators, cache
folders and notebook error outputs. It does not download organizer data;
training-only analyses must still be rerun locally before numerical changes
are reviewed. Inspect `git diff --cached --stat` before pushing a PR.

## Training-only physiology companion (review of PR #1)

The compact [C3/C4 mu-power notebook](notebooks/03_mu_power.ipynb) and
[reproduction instructions](analysis/mu_power/README.md) report the fixed
8–13 Hz task-to-baseline contrast for all 80 trials in each training run.
C3/C4 columns are looked up per file (see the channel-layout check below);
earlier outputs that used columns 5/9 for every run are superseded. The analysis never opens test
recordings and does not improve or select the decoder. Outputs are rebuilt
locally under ignored `results/mu_power/`; the notebook contains no saved
outputs. Windows include [2, 3.5) s and [2, 8) s after trigger; the cue is
at 2 s. Feedback onset is unknown, including for the shorter window.

## Riemannian decoder, decision-time curve and channel-layout check

```bash
stroke-rehab montage-check       # results/montage_check.csv
stroke-rehab compare             # results/model_comparison.csv
stroke-rehab decision-time       # results/decision_time.csv
stroke-rehab plot-decision-time  # results/decision_time.png (--theme dark for slides)
stroke-rehab lateralize          # results/lateralization.csv
stroke-rehab transfer            # results/transfer.csv
```

**Read this first.** Every test-run number in this section is exploratory.
The Riemannian model's settings were fixed in earlier solo work that had
already seen these test runs, and the team has inspected them too, so none of
these scores is an unbiased estimate. The 3.5 s line in the figure is the
feedback-phase start drawn in `DatasetInformation.pdf`. It is an assumption
from the protocol diagram, not a recorded FES timestamp: the files contain no
per-trial stimulation marker, so nothing before 3.5 s is shown to be
feedback-free.

### Same trials, same windows

`stroke-rehab compare` puts the selected filter-bank CSP model next to a
filter-bank Riemannian tangent-space model (seven 4 Hz bands from 4 to 32 Hz,
shrunk covariance, logistic regression, label-free online recentering; see
`stroke_rehab/riemann.py`). Both use the strict loader (80 trials per run), the
same seed-27 training folds, zero-phase filtering and one decision per test trial.
The CSP column reproduces `session_results.csv` and `timing_audit.csv` exactly.

| Window after trigger | Filter-bank CSP | Riemannian + recentering |
|:--|--:|--:|
| 2.5–6.5 s | 433/480 (90.2%) | 449/480 (93.5%) |
| Early, 2.5–3.5 s | 364/480 (75.8%) | 376/480 (78.3%) |
| Late, 4.5–6.5 s | 412/480 (85.8%) | 417/480 (86.9%) |

On 2.5–6.5 s the two models disagree on 38 trials (27 only Riemannian correct,
11 only CSP correct; exact McNemar p = 0.014, trials treated as independent,
which they are not across six runs). Training-run CV does not separate them
(442 versus 441 of 480), so a training-only selection rule would not reliably
pick the Riemannian model. Without recentering it scores 441/480. The earlier
solo figure of 444/479 used a 2.5–8.0 s window and dropped one P1 POST test
trial (its loader required two seconds of EEG before the trigger; both P1
POST runs start under 1.4 s before their first trial). With all 80 trials that
window gives 444/480.

### Accuracy against decision time

`stroke-rehab decision-time` uses **causal** one-pass filters and a one-second
window ending at the decision time, so no EEG after that time is used. Models
are fitted on the training run and scored on the test run every 0.25 s.
[Figure](results/decision_time.png); [dark version](results/decision_time_dark.png).

| Decision time | 2.0 s | 3.0 s | 3.5 s | 4.25 s | 6.5 s | 8.0 s |
|:--|--:|--:|--:|--:|--:|--:|
| Filter-bank CSP | 48.1% | 59.4% | 74.6% | 84.8% | 77.7% | 72.5% |
| Riemannian | 50.6% | 55.6% | 71.2% | 86.2% | 84.4% | 74.2% |

Both models are at chance up to the instruction at 2 s. A 65–95 Hz power
check found no consistent stimulation trace from which feedback onset could be
timed, so the interval before 3.5 s is labelled "early post-cue".

### Channel order differs between files

`montage.png` lists FC3 FCz FC4 C5 C3 C1 Cz C2 C4 C6 CP3 CP1 CPz CP2 CP4 Pz.
The bundled paper lists FC5 FC1 FCz FC2 FC6 C5 C3 C1 Cz C2 C4 C6 CP5 CP1 CP2
CP6. Neighbouring electrodes share more signal than distant ones, so
`stroke-rehab montage-check` rank-correlates electrode distance with signal
correlation (8–30 Hz, median over trials; no labels are used) for each layout.

| Files | Fit to `montage.png` | Fit to paper layout | Mains notch |
|:--|--:|--:|--:|
| P1 POST (2 files) | −0.96, −0.97 | −0.22, −0.24 | 60 Hz |
| P1 PRE (2 files) | −0.13, −0.12 | −0.81, −0.80 (−0.93 with columns 13/14 exchanged) | 50 Hz |
| P2, P3 (8 files) | −0.15 to −0.23 | −0.88 to −0.93 | 50 Hz |

- **Only the two P1 POST files follow `montage.png`.** They also carry a
  60 Hz notch where all others have 50 Hz, so that session used a different setup.
- **The other ten follow the paper's layout.** In both P1 PRE files, columns
  13 and 14 (CP5, CP1) fit clearly better exchanged; C3 and C4 are unaffected.
- **Order, not only electrode set:** with the layout assigned to each file,
  none of the 120 possible two-channel swaps improves the fit in any file.
- **What this cannot show:** a left-right mirrored layout fits identically. In
  every file the lowest-amplitude channel is on the right edge, which agrees
  with the paper's right-earlobe reference but does not prove orientation. The
  mapping is inferred from the signals and is not confirmed by the organizer.

Within-session decoding does not depend on channel names. Anything that names
a hemisphere or moves a model between P1 sessions does: **C3/C4 are columns
7/11 in ten files and 5/9 only in P1 POST.** `stroke_rehab.montage.layout_for`
returns the layout per file, and `analysis/mu_power/mu_pass.py` now uses it.
The C3/C4 mu-power outputs produced earlier with columns 5/9 read FC6 and Cz
in five of the six training runs and should not be used in slides.

`stroke-rehab lateralize` uses that mapping, causal 13–30 Hz power relative to
0.5–2 s, and C3/C4 each minus the mean of their two row neighbours (present in
both layouts). In the early window (2–3.5 s) no patient shows a PRE-to-POST
change in contralateral-minus-ipsilateral power (all Mann–Whitney p > 0.07).
The PRE-to-POST hemisphere flip for P1 in the earlier solo analysis came from
applying the `montage.png` labels to P1 PRE and does not hold.

### Cross-session transfer with name-matched channels

`stroke-rehab transfer` fits the Riemannian model on the **training run only**
of the same patient's other session and applies it to the target session with
no calibration data from that day (2.5–6.5 s, zero-phase filters). Each run is
recentered on itself without labels; the target is recentered causally from
its first trial. `common_names` uses the ten electrodes present in both
layouts (FCz C5 C3 C1 Cz C2 C4 C6 CP1 CP2), matched by name;
`columns_as_stored` uses all 16 columns in file order.

| Target test run | Common names (10 channels) | Columns as stored (16) |
|:--|--:|--:|
| P1 PRE | 68/80 | 56/80 |
| P1 POST | 73/80 | 53/80 |
| P2 PRE | 64/80 | 67/80 |
| P2 POST | 68/80 | 69/80 |
| P3 PRE | 72/80 | 73/80 |
| P3 POST | 72/80 | 75/80 |
| Pooled | 417/480 (86.9%) | 393/480 (81.9%) |

For P1, transfer is near chance with columns as stored and recovers when
channels are matched by name, which is consistent with a channel-order
mismatch between the two sessions; other differences between those recordings
(such as the different recording setup) are not ruled out. For P2
and P3 the layout is the same in both sessions, so the 16-column result is the
valid one. Three of the six targets are decoded with a model from the LATER
session, which could not happen in practice. Scores on the target training
runs are in the CSV.
