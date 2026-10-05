# Overnight verification and research

Authorized by Jason on 4 October 2026. Goal: a reviewable package by 08:00 London on 5 October, stopping earlier when its outputs are complete, with a maximum six hours of active assistant work. This is assistant work, not a calendar booking for Jason.

The isolated `analysis/overnight-verification` worktree starts with the accuracy/calibration study `c1a6d4`, incorporating Anna's current main `45d5b91` as merge `1edae10`. Existing mu PR #1 and the original accuracy worktree are preserved.

For independent review, the model/research files are packaged on `analysis/calibration-benchmarks`, based directly on main `483ac87`. This branch omits the physiology changes already in PR #1. The receipts retain the actual combined-worktree snapshots on which they were produced. The artifact checker executes whichever notebooks are present, including the physiology notebook after PR #1 is incorporated.

Jason subsequently asked to continue using the newest commits. Fetch and compare remote main before every new batch and the final handoff. Record the exact snapshot for each result. Complete a running frozen comparison before integrating new source; rerun only checks affected by relevant upstream changes. The final integrated fetch points to `483ac87`. This adds README/preflight changes and leaves the reproduced numerical analysis unchanged.

Anna's message to the other collaborator asks them to lead the time-resolved decoder comparison and accuracy-versus-decision-time figure. This review supports their work with reproduction, montage verification and evaluation checks; it does not establish that they accepted every requested deliverable.

## Outputs and finish condition

1. Compare frozen numerical outputs for inventory, baseline, timing audit, diagnostics, causal replay, nested selection, latency audit and the latest negative controls; verify local calibration export and notebooks separately. Record source/runtime/input hashes. Report mismatches as mismatches.
2. Review primary papers and distinguish supported findings from proposed work. [Literature review](LITERATURE_REVIEW.md).
3. Audit the existing calibration comparison and, if justified, make at most one training-only split sensitivity analysis, specified before evaluating. Preserve original results and do not rescore exposed tests for new choices.
4. Save a concise report and practical team handoff, with validated code in a draft PR if appropriate. No main merge, human message, organizer email, raw EEG/model upload, paid compute or clinical claim.

## Reproduction

Use an isolated Python 3.11+ environment and install `.[accuracy,verification,test]` for all optional model/notebook checks. The repository's default installation does not require pyRiemann; its optional model tests skip when that dependency is absent.

```text
python analysis/overnight/reproduce.py --data-dir /path/to/local/stroke-rehab
python analysis/overnight/verify_artifacts.py --data-dir /path/to/local/stroke-rehab
python analysis/overnight/selection_sensitivity.py --data-dir /path/to/local/stroke-rehab
```

The original data are read in place. Regenerated outputs, logs and fitted models go under ignored `data/overnight-reproduction`; the small receipt goes to `results/overnight/reproduction.json`. Reproduction of already fixed models using exposed test runs does not create a fresh held-out evaluation. Notebook runs and scientific interpretation remain separate checks.

## Montage check

Visual inspection of the supplied `montage.png` gives the ordered labels:
`FC3 FCz FC4 C5 C3 C1 Cz C2 C4 C6 CP3 CP1 CPz CP2 CP4 Pz`.
This matches the reference repository's `mi.CH`. The mu implementation uses
zero-based C3/C4 indices 4/8, corresponding to montage numbers 5/9.
This verifies consistency with the supplied montage; it does not recover
participant-specific electrode coordinates, independently certify hardware
connections, or establish the acquisition reference.

The reference `mi.py` requires [-2,8) seconds around each trigger and excludes
one boundary trial in each P1 POST run. Anna's corresponding analysis retains
80 trials per run. Comparisons need identical retained trial IDs or explicit
79/80 counts. Actual per-trial FES timestamps remain unavailable; an early
post-cue interval must not be called definitely feedback-free.

## Completed outputs

[Morning report](../../results/overnight/MORNING_REPORT.md) summarizes nine exact pipeline CSV matches, three executed notebooks, six verified local model exports, the five-paper review and the single training-only split sensitivity. The newer preflight commit does not alter numerical methods. Both original and sensitivity results are retained; no new exposed-test scores were calculated for the sensitivity. Anna physiology PR #1 is ready for review at `8f44f8d` after 42 tests and preflight passed.

## Follow-up integration — 5 October 2026

Main advanced to `44d240f`: Anna carried over the same physiology module, tests and all eight notebook code cells while removing saved outputs. PR #1 is superseded by that commit. PR #2 retains the accuracy/verification extras and adds Anna's notebook extra and Python ≥3.11 requirement. Raw data and generated physiology outputs remain ignored on this branch; main itself has no tracked ignore file.

The artifact runner now requires all three frozen physiology CSV hashes in `mu_reference_hashes.json`; an absent reference directory cannot silently produce zero checks. The current compact main notebook executes eight code cells without errors and matches all three historical CSV hashes, with unchanged six training input hashes. See `main_integration_check.json`. Earlier pipeline/export receipts keep their original snapshots.

The supplied-montage mapping remains an assumption for each recording. PR #3's correlation topology does not verify acquisition labels or orientation; it proposes a layout hypothesis. Cross-run transfer and source pooling assume channel correspondence. PR #3's adapted Riemannian pipeline, feedback plots and transfer contract differ from this fixed-reference comparison; do not pool their scores.

Latest collaborator snapshot checked: `a3382b7`. It fixes auxiliary transfer to fit the source training run only and changes its physiology to a per-file correlation-inferred layout, adding a CP5/CP1-swap hypothesis. The source-label concern is resolved in that branch. Other contract differences remain: inferred acquisition mapping, zero-phase filtering, unlabeled recentering and both temporal directions. This is a source review of that snapshot, not a reproduction of its changed numerical outputs.


## Current integration — main dbab3b7, 5 October 2026

PRs 3 and 4 are now merged. The current physiology module uses inferred
per-file channel mappings, so the fixed-montage CSV hashes above are historical.
The artifact runner now rejects changed physiology source before execution;
it cannot overwrite an old receipt or claim parity under a different map.
The old numerical receipts remain valid for their recorded source snapshots.
Run the current mu module separately for current descriptive outputs; those
outputs are not a reproduction of the historical fixed-montage CSVs.

PR2 was closed automatically when its head branch was renamed. Its replacement
review branch is `analysis/calibration-benchmarks`, reconciled with main dbab3b7.
The source-level channel map and ERD/ERS limitations are in `docs/protocol.md`.
New chronological selection and covariance-stream work have separate branches
and receipts; their results must not be pooled with this four-second benchmark.
