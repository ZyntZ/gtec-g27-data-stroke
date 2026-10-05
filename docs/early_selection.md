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
CSP. At budget 10 there is no inner validation: use the declared CSP fallback.
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
Three participants supply six repeated sessions; 120 pooled trials are not
120 independent people. Keep P2/P3 failures visible.

Run from the repository root:

```sh
python -m stroke_rehab.early_selection --data-dir /path/to/stroke-rehab
python -m pytest -q
```

The fixed design is in `configs/early_selection.json`. Derived outputs are in
`results/early_selection/`: session scores, outer/inner predictions, choices
and input/source/runtime hashes. Changing the configuration creates another
exploratory analysis, not a continuation of the frozen result.
