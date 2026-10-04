# Evaluation contract

## Source and decoding target

The official archive contains 12 MATLAB MAT recordings: patients P1–P3 × first (`pre`) / last (`post`) rehabilitation session × `training` / `test` run. Each file has `fs`, `y` (sample × 16 EEG channels), and `trig`. Inspection of all 12 files found 256 Hz, 80 eight-second labelled trigger blocks per run (40 `+1`, 40 `-1`). The source `DatasetInformation.pdf` defines `+1` as left-hand motor imagery, `-1` as right-hand motor imagery, and says the instruction comes two seconds after the trigger. The labels repeat for the whole 2048-sample block; treating all labelled samples as independent examples would catastrophically inflate the sample count. `results/data_inventory.csv` includes a checksum for each original MAT file.

**Target:** binary, one decision per test trial. **Unit of independence:** trial, not individual samples or sliding windows. We do not interpret PRE/POST differences in three participants as a treatment effect.

## Locked pipeline for this update

1. Read each MAT file and verify sampling rate, finite EEG, 16 channels, 80 unbroken blocks, expected block length, and both classes.
2. Apply a zero-phase, fourth-order Butterworth bandpass filter to the *continuous run*, separately for 8–12, 12–20 and 20–30 Hz. Filtering is offline and noncausal; do **not** call this an online-ready decoder.
3. Take post-instruction windows at 2.5–3.5, 3.5–4.5, 4.5–5.5 and 5.5–6.5 s relative to the trigger. Baseline A uses log channel variance per band/window, a training-fitted standardizer, then shrinkage LDA. Baseline B learns trace-normalized, ridge-stabilized CSP spatial filters on the training trials, uses two eigenvectors from each end per band, computes log relative spatial variance over 2.5–6.5 s, standardizes on training trials, then fits shrinkage LDA.
4. For each patient/session, use *only its training run* for a seeded, shuffled, stratified **5-fold trial-level** CV. CSP, scaler and LDA are fitted from scratch within every fold. Select the model with greatest out-of-fold trial accuracy (fixed candidate order breaks ties); refit on all 80 training trials. Only then extract test features and make predictions on that session's 80-test-trial run. Test labels are used solely to calculate the final score. No pooled cross-session training, no test-run-based hyperparameter selection.
5. Report exact `correct / 80`, accuracy and balanced accuracy. Classes are balanced, but balanced accuracy is kept for future checks. The pooled 433/480 describes these recordings only; it is not a confidence interval over patients.

The power baseline and CSP were fixed before examining test outcomes within the scripted experiment, but this is an exploratory hackathon dataset: subsequent human choices may have been informed by seeing test data. Any future model chosen with knowledge of these test outcomes needs *new subjects or a new genuinely held-out run* for an unbiased performance claim. Out-of-fold training accuracy is the only honest score for choosing improvements on this release.

## Timing sensitivity

`stroke-rehab audit` fits the same fixed CSP pipeline separately on 2.5–3.5 s and 4.5–6.5 s windows. It does **not** replace the selected main model, and no best window is selected based on test performance. The original dataset documentation says training delivers visual + functional electrical stimulation (FES) feedback, while test feedback is contingent on correct decoding. Late test windows can therefore carry consequences of earlier correct/incorrect detection. This creates a feedback confound; late-window offline accuracy should not be sold as a pure pre-feedback intention-decoding score. The early window is a diagnostic, not guaranteed to precede every feedback event: precise feedback timestamps are not included.

## Timing and physiology guardrails

The original offline timing audit filters the complete continuous recording
in both directions before cutting an epoch. Samples after the decision can
therefore affect its early-window features. It is not a causal or necessarily
feedback-free benchmark. The separate one-pass causal replay addresses the
backward-filtering issue, but missing per-trial feedback timestamps still
prevent a feedback-free interpretation.

The training-only [mu-power protocol](../analysis/mu_power/README.md)
compares within-trial spectral power; it is not a classifier score. PRE and
POST are repeated recordings from three participants, not six independent
patients or proof of improved clinical outcomes.

## How these metrics differ from the organizer's table

The organizer reports `CSP+LDA` and `PCA+TVLDA` accuracies in `overview.pdf`. We report one fixed-window prediction per trial from a different model and implementation. Their referenced paper describes averaging classification accuracy over multiple time steps, then taking a maximum over a feedback period. We have **not** reproduced that protocol and do not claim our scores improve on it. Nor can offline `sosfiltfilt` be put directly into a real-time stimulator.

## Sources bundled with the organizer archive

- `DatasetInformation.pdf` (trigger timing, train/test feedback semantics).
- `overview.pdf` (MAT variable descriptions and the original method table).
- `StrokeRehab.pdf` (feasibility study; interpretation of treatment and BCI feedback).
- `Gruenwald et al. - 2019 - Time-Variant Linear Discriminant Analysis Improves.pdf` (methodological background, not our implemented classifier).

## Added training-only controls (not a replacement leaderboard)

`stroke-rehab diagnose` loads only the six training MAT files. Its JSON
manifest `configs/diagnostics.json` fixes the five folds, seed 27 and
one-neighbour purge. It compares random stratified trial folds with five
contiguous 16-trial validation blocks. The latter exclude adjacent trials
from each training partition but score each trial once. Every fitted CSP,
scaler and LDA is learned anew on that fold's training trials. These scores
do not select or modify the original held-out models. The post-cue CSP
uses the original offline 2.5–6.5 s feature and is **noncausal**. The other
control is causal 8–12, 12–20 and 20–30 Hz channel log-power, with a
0.5–1.75 s window entirely before the instruction at +2 s. Because it is
filtered causally on the continuous run, later EEG cannot enter this window.
Its analysis still cannot determine when feedback begins.

| Training run | Pre-cue random | Pre-cue blocked | Post-cue CSP random | Post-cue CSP blocked |
|:--|--:|--:|--:|--:|
| P1 PRE | 46/80 | 39/80 | 77/80 | 75/80 |
| P1 POST | 33/80 | 31/80 | 79/80 | 79/80 |
| P2 PRE | 35/80 | 34/80 | 71/80 | 71/80 |
| P2 POST | 37/80 | 33/80 | 74/80 | 74/80 |
| P3 PRE | 30/80 | 31/80 | 72/80 | 72/80 |
| P3 POST | 36/80 | 36/80 | 68/80 | 68/80 |

Do not interpret the pre-cue numbers as proving a clean decoder: some are
below 50%, and inverted classification can itself be informative. Neither
these 80 trials nor the six runs are independent patient replications.
Temporal folds can still share stable within-run artifacts. The notebook's
`gross_excursion_trials` flags a trial if at least one channel has post-cue
peak-to-peak amplitude >5× that channel's within-run median. This is a
descriptive, unit-free heuristic, not an automated exclusion or clinical
artifact definition. There are 2 flagged training trials in P1 PRE and
2 in P2 POST, 0 in each other training run. The original baseline has not
been reranked or retrained using these diagnostics.

## Fixed causal training-only replay

`configs/causal_training.json` defines a 64-sample (~250 ms) chunk, a
2.5–3.5 s post-trigger decision window, five chronological validation
blocks and one adjacent trial removed on each training side. Each session
has a separate filter state, reset at the beginning of its continuous run.
Incoming triggers are supplied only when their chunk arrives. A fourth-order
Butterworth 8–12, 12–20 and 20–30 Hz filter bank operates with stateful,
one-pass `sosfilt`; centred log variance per channel and band is accumulated
incrementally. Training partitions fit a standard scaler and shrinkage LDA
from scratch. `stroke-rehab causal-train` loads the six training MAT paths,
never reads any test MAT and never selects a model using known test labels.
The fitted full-training model is used only to benchmark end-to-end replay;
its in-sample predictions are not scored. `results/causal_train_cv.csv`
stores deterministic out-of-fold scores and decision times in EEG seconds.
Runtime measurements are printed to the console rather than committed since
they depend on the executing CPU. A decision stamped with the end sample of
its chunk is not an experimental feedback timestamp or a measurement of
external device latency. The data do not contain timing of contingent FES.

The causal short-window power baseline is intentionally small (48 features,
80 training trials per session) and has no window or hyperparameter search.
With purged chronological folds, performance varies widely across sessions:
59, 70, 35, 37, 48 and 36 correct out of 80, respectively. The poor scores
for P2 and P3 must not be hidden by pooled accuracy or by relabelling this
model a rehabilitation decoder. A model using earlier training trials can
still inherit filters from previously seen validation EEG through the
continuous, causal filter state. One full-trial purge makes that remote
filter-state dependence small but does not establish sample independence.
The interleaved per-trial triggering and zero initial filter state are
algorithmic choices, not claims about the organizer's exact online hardware.

## Nested causal covariance protocol

Versioned configuration `configs/spatial_nested.json` fixes the bands, the
2.5–3.5 s window, 64-sample chunk size, five chronological outer folds,
three chronological inner folds, one purged neighbouring trial, two filter
pair counts (1 and 2), and ridge 0.01. The control is channel log variance,
computed from the same stream's covariance diagonals; this ensures equal EEG
availability and filter state for all candidates. The spatial estimator uses
trace-normalized class-average covariances, ridge-stabilized generalized
eigenvectors fitted on only the current fold's calibration trials, and log
relative projected variances. All 80 training trials receive one outer-fold
prediction per fixed method; the nested rule selects only by inner training
accuracy (power first for exact ties). Validation trial labels are read only
to score their own outer fold. Every programmatic input path ends in
`_training.mat`; a calibration function rejects `_test.mat` paths. A local
calibration model is trained on all 80 training trials after selection by
training-only CV, never on held-out runs.

| Training run | Nested | Power | CSP 1 pair | CSP 2 pairs |
|:--|--:|--:|--:|--:|
| P1 PRE | 66/80 | 59/80 | 70/80 | 66/80 |
| P1 POST | 77/80 | 70/80 | 76/80 | 78/80 |
| P2 PRE | 40/80 | 35/80 | 46/80 | 43/80 |
| P2 POST | 48/80 | 37/80 | 46/80 | 49/80 |
| P3 PRE | 55/80 | 48/80 | 56/80 | 53/80 |
| P3 POST | 43/80 | 36/80 | 46/80 | 47/80 |

The within-run outer-fold scores are an estimate for the predefined
algorithm-selection rule, **not** an independent estimate for a new decision
to choose CSP-1 after viewing this table. The 480 trial outcomes are
clustered within just three patients and six runs. Purged blocks reduce,
but cannot eliminate, within-run dependence. The causal filter's state can
carry earlier EEG into later segments, although each trial is >8 s long;
there is no selective refit on a test recording. External stimulation and
feedback timestamps are not available. A classifiable early-window signal
may be imagery, unintended cue information, or feedback. Model files use
joblib/pickle and are unsafe to load from untrusted sources. This research
replay never actuates functional electrical stimulation.

## Fixed-model decision-latency sensitivity

`configs/spatial_latency.json` fixes a single 1-pair CSP and 2.5 s window
start while ending its observation at 3.5, 4.5, 5.5 or 6.5 s after trigger.
`stroke-rehab latency-audit` fits each method only on the corresponding
training portion of the same five chronological, one-neighbour-purged folds.
`results/causal_latency_audit.csv` contains six sessions × four fixed
endpoints. The output is a feedback-timing *diagnostic*, not a competing
nested learner or permission to optimize against its validation labels.
The aggregate counts 340/480, 384/480, 418/480 and 436/480 do not show
motor-intent accuracy improves: later training trials can contain visual
and functional electrical stimulation (FES) feedback. Subject-wise curves
and numeric values are reproduced by `notebooks/02_causal_latency.ipynb`.
The fixed early window remains the calibration default until feedback
timestamps and the organizers' scoring contract establish a valid endpoint.
