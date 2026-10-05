# G27 pull request reliability review

Reviewed 5 October 2026. The integration combines main `07d725f`, PR6
`e5f8898`, PR7 `4b72159` and the updated PR8 `70c504f`. Passing individual
branches and matching normal predictions did not establish that combined
source receipts or failed streaming decisions were safe to reuse.

| Finding | Cause and response |
|:--|:--|
| PR5 advances state before a prediction returns | A classifier exception changes reference/count without delivering a decision. PR8 computes the decision before committing state and rolls back a failed covariance batch. |
| Original PR8 cannot safely retry failed EEG decisions | Classifier rollback leaves consumed EEG/filter/sample state advanced. Updated PR8 makes the decoder terminal after any processing error. Recovery reconstructs the decoder and replays from a known run boundary. |
| Source receipts drift across PR7/8 | PR7 fingerprints old Riemannian source; PR8's artifact test ignored extra stream-source fingerprints. Actual reruns preserve the old snapshot and reproduce predictions/input hashes. The strengthened guard verifies every declared current source, output hash and prediction/correctness consistency. |
| PR6 needs clearer evidence labels | Random balanced label subsets are not chronological calibration; exposed-test 446/480 is exploratory. Figure/report captions are corrected and historical receipts are distinguished from the integration checks. |

An independent review was requested with `gpt-6-astra` at Ultra effort and completed after a capacity retry. Its runtime identity is not independently exposed by agent metadata. The review inspected published sources and Anna's comments; this checkout executed the checks.

## Current-main integration — 5 October 2026

Anna's newer comments on PR6/7/9 identify current-main update03 as the remaining
integration dependency. Main `0413792` and PR9 `173a3a3` were combined with PR6;
source repair `5037fab` then received an actual organizer-metric rerun and stream
replay. All three organizer CSV hashes and all twelve raw input hashes are
unchanged. The full combined suite passes **136 tests**, with optional accuracy
dependencies installed, plus preflight and all 19 published-arithmetic checks.
Exact commands/source fingerprints are in `verification.json.current_stack_checks`.

The organizer manifest now requires all six calculation sources and all four
outputs, with source hashes normalized across Windows/Linux text checkouts.
The stream manifest also requires its reproduction driver, spatial operations
and event definitions. Both active receipts are produced by execution, with
their preceding snapshots preserved. PR7 adopts only shared repairs plus current
main, checks identical executed dependencies/outputs, and passes 111 component
tests. PR5/8 remain one integration unit; the unchanged PR5 base is not separately
safe to merge. The one-page PDF and earlier hosted pass remain historical.

The new organizer rerun reads the already-inspected test MAT files; the forward
replay remains training-only. These distinctions replace any broad claim that
the whole current integration reads training data only. No model search or
independent accuracy improvement follows from the unchanged numerical results.

## Historical checks and interpretation

The source fingerprint failure was reproduced before repair. The current
verification commands and exact execution evidence are recorded in
`results/pr_reliability/verification.json`. Local verification passed 127 tests with one optional module skipped. GitHub verified all 130 tests with the optional accuracy dependencies, plus preflight and the published-arithmetic check on `d1a84f2`. Automated pull-request, push and
merge-queue checks run the full data-free suite, repository preflight and
published-result arithmetic validator. They do not download participant EEG
or fit new clinical models. A workflow passing on one branch does not imply
that an untested combination passes; receipts must match the executed sources.

The early result remains **CSP 82/120 versus selected 75/120** at 60 labels
and +3.5 seconds. Retrospective exposed tails, uncertain feedback onset and
three participants limit interpretation. These fixes improve implementation
and reproducibility; they do not establish independent accuracy improvement,
motor-intent specificity or rehabilitation benefit.

## Evidence and review order

[Anna on PR5](https://github.com/ZyntZ/gtec-g27-data-stroke/pull/5#issuecomment-5989971381),
[PR8 failure handling](https://github.com/ZyntZ/gtec-g27-data-stroke/pull/8#issuecomment-5989985404),
[PR7 combined receipt](https://github.com/ZyntZ/gtec-g27-data-stroke/pull/7#issuecomment-5989993987),
[PR6 reporting](https://github.com/ZyntZ/gtec-g27-data-stroke/pull/6#issuecomment-5990009998).
Review this integration against PR6 before choosing how to merge PR5/7/8.
No main merge or competition submission is implied.
