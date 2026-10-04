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
