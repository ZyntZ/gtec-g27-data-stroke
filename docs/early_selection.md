# One fixed early decision with chronological model selection

This extends the merged forward-calibration audit with a **selection policy**.
It is a retrospective analysis of the six training recordings. Their later
tails have already been examined; these results are not independent validation.
No test recording is opened and no model is chosen using a tail/test score.

The frozen endpoint is **+3.5 seconds after the attention trigger**, using
the causal interval **[2.5,3.5)**. The imagery cue is at +2 s. This is early
post-cue EEG, not certified feedback-free EEG. Nominal feedback starts around
+3.5 s; actual per-trial FES timestamps are unavailable. Sample cutoff and
delivered device latency are different quantities; delivery is not measured.

For each session and budget 10/20/40/60, calibration is the first `n` trials.
The fixed scored tail is **trials 61–80**, with 20 decisions at the same time
for every method/budget. Selection compares two existing fixed pipelines:
filter-bank CSP with shrinkage LDA and adaptive Riemannian tangent features.
They use different band banks, so this is a complete-pipeline comparison.

Inside the calibration prefix, fit trials 1–10 and validate 11–20, then fit
1–20 and validate 21–30, continuing in ten-trial blocks only up to the budget.
Pool those forward predictions and choose by balanced accuracy. Ties favor
CSP, using exact integer class-count cross-products rather than rounded
floating-point scores. Exact ties have a separate saved selection reason.
At budget 10 there is no inner validation: use the declared CSP fallback.
Skip an inner fold with a single-class fit; use the same fallback if pooled
validation lacks a class. Save every fold, score, choice and prediction.
Refit the chosen pipeline on its complete allowed prefix, then score the tail
once. The chosen-policy tail score does not affect the choice.

The reference stream resets at the start of each validation block. Gap trials
between calibration and trial 61 do not update the Riemannian reference;
continuous filters still carry their earlier EEG. This matches the merged
audit, rather than silently changing the adaptation protocol. Calibration
majority (ties choose −1) is an additional reference for class imbalance.
Report both class recalls, balanced accuracy and integer correct/20 per run.
The summary also includes the predicted-left fraction and agreement with the
calibration-majority label, making single-class decisions visible.
Three participants supply six repeated sessions; 120 pooled trials are not
120 independent people. Keep P2/P3 failures visible.

Run from the repository root:

```sh
python -m stroke_rehab.early_selection --data-dir /path/to/stroke-rehab
python tools/plot_early_selection.py
python -m pytest -q
```

The fixed design is in `configs/early_selection.json`. Derived outputs are in
`results/early_selection/`: session scores, outer/inner predictions, choices
and input/source/runtime hashes. Changing the configuration creates another
exploratory analysis, not a continuation of the frozen result.

## Frozen result and limits

At 60 calibration trials, CSP gets **82/120** tail decisions correct and the
training-selected policy gets **75/120**. Selection provides **no pooled
accuracy gain**. The six shared-axis panels in
`results/early_selection/early_calibration.svg` (also PNG) retain all four
strategies and the same 20-trial tail at every budget. Balanced accuracy is
the mean of the two class recalls; the constant-majority reference is 0.5.
Pooled correct counts describe trials, not independent participant evidence.

Of the 24 session/budget choices, six use the budget-10 fallback, four are
exact inner-score ties, ten are strict CSP wins and four are Riemannian wins.
All Riemannian choices occur in P2. In P2 PRE at budget 60 the inner balanced
accuracy advantage is only **1/624 (about 0.00160)**; both pipelines get
25/50 pooled inner decisions correct. This is a small class-weighting margin,
not a robust model advantage. Nine of 48 family/session/budget cells predict
only one class; the prediction-fraction and majority-agreement fields expose
these outcomes without changing any model or decision rule.

The inner and final horizons differ. Inner folds fit smaller prefixes and
score the immediately following ten trials; final fits use the entire budget
and score trials 61–80, potentially after a gap. Inner balanced accuracy does
not estimate that same final horizon. Addressing this needs a separately
declared future protocol, not retuning on the exposed tail. The +3.5 s endpoint
and CSP default were frozen after earlier audit results were available, so
this is a **retrospective procedural audit**, not a preregistered or prospective
claim. Offline predictions cannot establish user learning or rehabilitation.

## Reproduction receipts

`early_provenance.json` records the exact design, six training-file hashes,
all prediction/summary/choice output hashes, runtime and source dependencies
(including both feature banks and the forward-calibration helper). Source
hashes normalize CRLF to LF; CSV hashes identify exact bytes, protected by
`.gitattributes`. The recorded Git commit is the execution base; the source
hashes identify the actual executed working-tree files. The receipt test
reconstructs choices and summaries and matches every one of the **960**
fixed-time family predictions against the merged forward audit, per trial.
Budget 10 alone writes a valid inner-prediction CSV with headers and no rows.

The pipeline causality/reset test uses generated signals only: changing EEG
at or after the last prefix decision cutoff cannot change earlier features
or that prefix's choice, and repeated batch predictions reset their adaptive
reference. A later signal may affect later trials through the continuous
filter state; this is intentional and is not a claim of trial independence.
