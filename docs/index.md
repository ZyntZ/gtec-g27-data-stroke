# Documentation map

[Home](../README.md) · [Evaluation contract](protocol.md) · [Experiment guide](experiments.md)

**Read the score definition before quoting a percentage.** The organizer permits feedback-period EEG and published reference accuracies use a peak of average trial correctness over time. Our causal, fixed-time and training-only checks investigate different questions. None is a prospective clinical outcome: there are only three participants, and the supplied test recordings have already been inspected.

## Choose a page

| If you want to… | Go to… | Status |
|:--|:--|:--|
| Understand subjects, triggers, time, metrics and prohibited claims | [Evaluation contract](protocol.md) | Current shared rules |
| Re-run an analysis or find its output and interpretation | [Experiment guide](experiments.md) | Current navigation |
| Inspect calibration-budget and latency curves | [Forward calibration](forward_calibration.md) | Training-only, exposed tails |
| Audit early model selection, tie breaks and P2 failure | [Early selection](early_selection.md) | Training-only, no benefit over fixed CSP |
| Understand organizer peak scoring and its time grid | [Organizer metric](organizer_metric.md) | Previously inspected test runs |
| Compare training-chosen decision times to test scores | [Locked time](locked_time.md) · [Training-locked endpoint](training_locked_endpoint.md) | Different endpoint-selection rules; test scores exploratory |
| Verify adaptive Riemannian state and error handling | [Riemannian stream validation](riemann_stream_validation.md) | Implementation parity, not validation on new patients |
| Investigate stacked PR provenance and historical integration | [PR reliability review](pr_reliability_review.md) | Dated historical receipts, **not** a current test count |
| Review older calibration/transfer work | [Historical accuracy protocol](../analysis/accuracy_calibration/PROTOCOL.md) · [HTML report](../results/accuracy_calibration/REPORT.html) | Separate models/windows; retrospective |

## Scorecard

These saved numbers answer **different** questions. *Training* means the six training MAT files; *test* means the six paired public MAT files that have been examined during development. A numerator counts correct decisions, not independent patients.

| Pipeline / decision rule | Source and scored trials | Correct | How to interpret it |
|:--|:--|--:|:--|
| Fixed causal CSP, +3.5 s, 60-label fit | Training trials 61–80 in each of six sessions | **82/120** | Fixed early diagnostic; same trial IDs as the next row. |
| Fixed causal CSP, +4.25 s, 60-label fit | Same 120 training trials | **99/120** | Extra time may add feedback-related signal. |
| Forward-validation-selected early family | Same 120 training trials; +3.5 s | **75/120** | Selection did **not** improve on fixed CSP. |
| Chronological, repeatedly refitted CSP at +3.5 s | Training trials 21–80 in each session | **245/360** | Fits only prior labels, but distinct protocol and trial count. |
| Original offline CSP/power baseline | Test trials 1–80 in each session | **433/480** | Noncausal filtering, later EEG, exposed test; not the peak organizer method. |
| Training-selected full-window endpoint | 480 public test trials, different chosen endpoint per session | **388/480** | Training-chosen per-session time, not a single early cutoff. |
| Organizer-style fixed +3.5 s curve | 480 public test trials, 80-label model fits | **358/480** | Descriptive fixed-time score. |
| Globally training-locked +4.25 s curve | Same 480 test trials, 80-label model fits | **407/480** | One endpoint chosen using training tails; test is still exposed. |
| Sum of organizer-style session-specific best times | Same 480 test trials | **427/480** | Six times chosen *post hoc* on test labels, not deployable prospective accuracy. |

Do **not** contrast 433/480 with 358/480 as a drop caused *only* by earlier decisions: filtering, feature extraction and model fits also differ. Do **not** quote the 427/480 test peak as the accuracy of one fixed-time online decoder. Consult [the complete trial-level definitions](protocol.md) and [the organizer timecourse](../results/organizer_metric/organizer_metric_timecourse.csv).

## What remains unknown

- Feedback onset is not logged per trial; +3.5 s in the figure is an approximate diagram boundary, not a functional electrical stimulation (FES) timestamp. Earlier EEG is not certified feedback-free.
- Channel order and orientation are not independently documented per MAT file; the per-file mapping comes from a correlation heuristic. Lesion hemisphere and affected hand are not supplied.
- Calibration tails and public test runs have been inspected repeatedly. Neither passing unit tests nor reproducing numerical CSVs creates a blind evaluation or proves rehabilitation benefit.

For a prospective claim, freeze the entire model, endpoint and failure policy **before** collecting independent participants or sessions and validate against feedback/artifact controls.
