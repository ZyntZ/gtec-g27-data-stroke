# Evaluation contract

[Home](../README.md) · [Documentation map](index.md) · [Experiment guide](experiments.md)

This contract separates **training-only diagnostics**, **previously inspected test-file evaluations** and **historical runs**. It is an analysis of research EEG, not a medical-device or clinical-efficacy study.

## Data and trial identity

The organizer supplies 12 MATLAB MAT files: P1–P3 × first (`pre`) / last (`post`) rehabilitation session × `training` / `test`. Each file has `fs`, `y` (samples × 16 EEG columns) and `trig`. The validated files have 256 Hz sampling and **80 labelled eight-second trial blocks per file**, balanced 40/40 across instructions. A trigger value `+1` instructs *left-hand imagery*; `-1` instructs *right-hand imagery*. It is **repeated across 2,048 samples of one trial**, not 2,048 independent labels. The visual left/right instruction occurs +2 s after the attention trigger. Source: `DatasetInformation.pdf` inside the official archive; file hashes: [`results/data_inventory.csv`](../results/data_inventory.csv).

Training and test are **different runs from the same patient and session**. The 25 treatment sessions are not provided: only the first and last are available. Trials/windows are not independent patients; six sessions come from **three** participants. Hand-instruction labels do not identify actual movement, lesion side, the affected hand, recovery or treatment response.

## Which evidence lane?

| Lane | Fitting and scoring | Why it exists | Limitation |
|:--|:--|:--|:--|
| Forward-only training tails | Fit first 10/20/40/60 labelled training trials; score **the same later trials 61–80** at each budget | Compare calibration needs, model families and timing without reading test MAT files | The tails have been explored and are not an untouched future cohort. |
| Chronological prequential controls | Fit only preceding training labels; score trials 21–80 sequentially | Test whether a proposed feature still predicts when new labels arrive over time | A different 60-trial evaluation set; model selection based on these scores would require a new holdout. |
| Purged blocked training cross-validation (CV) | Five contiguous blocks; exclude one adjacent training trial on each side | Compare fixed candidates and diagnose temporal correlations | Often uses *future* calibration trials in a fold; not online forward-only validation. |
| Paired test-run analysis | Fit only the session's training MAT; calculate one decision per test trial at a declared time | Descriptive comparison with organizer-style evaluations | All six public test runs have been inspected repeatedly; **not blind** after development. |
| Historical offline benchmark | Run-wide zero-phase filtering; several later windows and separate model selection | Reproduce the original 433/480 pipeline | **Noncausal** and not comparable to early streaming with only one variable changed. |

Models, band banks, decision times and label budgets differ between lanes. Always state the **recording source, trial IDs, window relative to trigger, filter direction, calibration budget, selection rule, and numerator/denominator** with a score. [Experiment commands and receipts](experiments.md).

## Two scoring rules: do not substitute one for the other

For test trial `i` and time-grid point `t`, let `C[t, i]` be 1 if the predicted instruction equals its label, otherwise 0. The organizer's team-supplied clarification specifies:

```python
organizer_peak = C.mean(axis=1).max()  # average over trials, THEN maximize across time
```

It **does not** specify averaging per-trial best times (`C.max(axis=0).mean()`). The grid, preprocessing, model and reference windows must also be declared before attempting a reproduction of the published methods. Feedback-period EEG is permitted by the organizer and all 80 training trials may calibrate the paired test-run model. The submitted deliverable is a video. [Organizer-metric reproduction and its 23-point grid](organizer_metric.md).

For a **fixed deployable-time decision**, declare a single cutoff (or a rule for choosing it from training data) *before evaluating test labels*: score one decision per trial at that cutoff. On the exposed test curves, fixed +3.5 s is **358/480**; the training-tail-selected global +4.25 s is **407/480**. In contrast, the six **post-hoc, test-label-chosen session peaks** sum to **427/480**. A separate purged-CV **per-session** training-time selector gives **388/480**; these are two different training selection rules. None constitutes independent test validation here. The published `overview.pdf` CSP+LDA and PCA+TVLDA reference scores are not reproduced exactly by our distinct decoders/time grids. [Side-by-side definitions](index.md#scorecard), [global time lock](training_locked_endpoint.md), [session-specific time lock](locked_time.md).

Balanced accuracy is the mean of the two recalls. `correct / trials` is the primary transparent count; do not report a pooled trial-level confidence interval as if 120 or 480 trials were independent participants.

## Filtering, feedback and delivery time

The instruction occurs at +2 s. The protocol diagram suggests a feedback phase around +3.5 s, but these MAT files have **no per-trial visual/FES timestamp**. Training includes visual and functional electrical stimulation (FES) feedback; in test, feedback is conditional on an earlier successful BCI detection. Consequently a late signal may partly decode a *consequence* of earlier decisions. The interval `[2.5, 3.5)` s is **early post-cue**, not certified pre-feedback. The cue-locked voltage control further challenges attribution to motor intent.

The historical baseline filters the **whole recording in both directions** (zero-phase), so features measured at an early time may use future EEG. By contrast, the one-pass causal power/CSP replay maintains continuous filter state and receives onset samples only when a chunk arrives. A feature window ending at +3.5 s with **64 samples/chunk** (256 Hz) can produce a decision as late as almost +3.75 s of recorded EEG. This is *simulated sample availability*, **not** measured acquisition/processing/display/FES latency. Streaming decoders fail closed after an EEG or prediction error; replay from a known recording boundary is needed before further decisions. The implementation never actuates stimulation. [Timing details](experiments.md#timing-and-streaming).

## Anatomy and scope of interpretation

The source PDF and montage image provide **different** 16-channel orders. The file-specific order used for channel-labelled analyses is supported by unlabeled signal-topology checks: P1 POST appears closer to `montage.png`; the other recordings appear closer to the bundled paper (with a CP5/CP1 exchange in P1 PRE). This is **a heuristic, not acquisition metadata**; mirrored left/right orientation cannot be distinguished from EEG correlation alone. Channel names are unnecessary for within-session decoding but critical for anatomical or cross-session interpretation. [Channel-layout checks and transfer controls](experiments.md#channels-and-transfer).

Affected-hand and lesion-hemisphere information is unavailable. Neither a trial's left/right instruction nor exploratory event-related desynchronization/synchronization (ERD/ERS) can establish the affected side. Avoid ipsilesional/contralesional, neuroplasticity or therapy-efficacy claims. Trial-wise instruction EEG may reflect visual cue, sustained imagery, visual or electrical feedback, or artifacts; signal-source attribution is unresolved. The existing tests establish **code invariants**, not a clinical validation.

## If a prospective claim becomes possible

Freeze the decoder, feedback and abstention policy, observation cutoff, sensor order, calibration protocol and statistics *before* seeing new sessions/people. Record instruction and stimulation onsets per trial; use controls matched for visual cues, time and artifacts. Report per-participant results, including failed sessions, and test once on independent participants or genuinely undisclosed runs. None of that independent evidence exists in this repository.
