# Experiment guide

[Home](../README.md) · [Documentation map](index.md) · [Evaluation contract](protocol.md)

**Run from the repository root.** Commands that open EEG need the checksum-verified organizer archive unpacked in ignored `data/stroke-rehab/` (`stroke-rehab download`). Saved CSVs/figures are supplied for review; re-execution needs those files and may take substantially longer than `pytest`. Outputs in `results/` are derived research artifacts, not raw EEG.

## Pick the right experiment

| Question | Command | Main saved result | Evidence lane |
|:--|:--|:--|:--|
| Are the official files intact? | `stroke-rehab inventory` | [`data_inventory.csv`](../results/data_inventory.csv) | Source QC; training + test metadata |
| Does a fixed offline baseline run? | `stroke-rehab run` | [`session_results.csv`](../results/session_results.csv) | Previously inspected test; zero-phase/noncausal |
| Is early post-cue different from pre-cue? | `stroke-rehab prequential` | [`prequential_summary.csv`](../results/prequential_summary.csv) | Training only; chronological 21–80 |
| Are CSP, power or cue-locked voltage distinctive? | `stroke-rehab mechanism-audit` | [`mechanism_audit.csv`](../results/mechanism_audit.csv) | Training only; same 21–80 per method |
| How much calibration and observation time help? | `stroke-rehab forward-calibration` | [`forward_calibration_summary.csv`](../results/forward_calibration/forward_calibration_summary.csv) | Training only; shared 61–80 tails |
| Does training-only family selection beat fixed early CSP? | `python -m stroke_rehab.early_selection` | [`early_summary.csv`](../results/early_selection/early_summary.csv) | Training only; same 61–80 tails |
| Do causal models survive temporally blocked validation? | `stroke-rehab causal-train`, `stroke-rehab nested-train` | [`causal_train_cv.csv`](../results/causal_train_cv.csv), [`causal_nested_summary.csv`](../results/causal_nested_summary.csv) | Training only; blocked folds |
| How sensitive are scores to later EEG? | `stroke-rehab latency-audit` | [`causal_latency_audit.csv`](../results/causal_latency_audit.csv) | Training only; fixed CSP |
| What does organizer-style peak scoring report? | `python -m stroke_rehab.organizer_metric --reference-csv results/decision_time.csv` | [`organizer_metric_aggregate.csv`](../results/organizer_metric/organizer_metric_aggregate.csv) | Previously inspected test; **post-hoc peak** |
| Can time be locked using training only? | `python -m stroke_rehab.locked_time` and `python -m analysis.training_locked_endpoint --output results/locked_endpoint/summary.json` | [`locked_summary.csv`](../results/locked_time/locked_summary.csv), [`summary.json`](../results/locked_endpoint/summary.json) | Distinct time-selection rules; test still inspected |
| Does run-specific Riemannian state agree with replay? | `stroke-rehab stream-check` | [`riemann_stream_check.csv`](../results/riemann_stream_check.csv) | Implementation parity, not test accuracy |
| Are anatomical maps/feedback timings confirmed? | `stroke-rehab montage-check`, `stroke-rehab feedback-check` | [`montage_check.csv`](../results/montage_check.csv), [`feedback_timing_check.csv`](../results/feedback_timing_check.csv) | Diagnostics; **neither question is resolved** |

Extra exploratory commands: `stroke-rehab controls`, `stroke-rehab cue-audit`, `stroke-rehab diagnose`, `stroke-rehab audit`, `stroke-rehab compare`, `stroke-rehab decision-time`, `stroke-rehab plot-decision-time`, `stroke-rehab transfer`, `stroke-rehab lateralize`. Use `stroke-rehab <command> --help` for actual flags. [Older calibration protocol](../analysis/accuracy_calibration/PROTOCOL.md) describes a **separate**, non-chronological analysis.

## One slide with matched trials

```bash
stroke-rehab forward-calibration   # expensive EEG-dependent rerun
python analysis/jury_tradeoff.py    # checks saved trial IDs/labels; writes PNG, SVG, CSV
```

The figure uses the fixed one-pair 8–30 Hz filter-bank CSP pipeline: first **60** labelled training trials are fitted separately per session, then **trials 61–80** are scored once per fixed one-second interval. The four intervals end at +2.0, +3.5, +4.25 and +6.5 s from attention trigger; the +2.0 s control is **before** the imagery instruction. Causal filtering is applied to the continuous run; all four times use the same 120 trial IDs and true labels. The script checks one decision per `(patient, session, time, trial)`, fixed budgets, label identity and correctness before drawing [`same_tail_tradeoff.png`](../results/jury_tradeoff/same_tail_tradeoff.png). Numerical table: [`same_tail_counts.csv`](../results/jury_tradeoff/same_tail_counts.csv). Full [design, model-family comparison and caveats](forward_calibration.md).

### Animated version

```bash
python analysis/decision_clock.py --data-dir data/stroke-rehab   # training runs only; recompute, then draw
python analysis/decision_clock.py                                # draw from the saved counts, no EEG needed
```

[`decision_clock.gif`](../results/decision_clock/decision_clock.gif) sweeps the same design through the trial: same model, same 60 calibration trials, same trials 61–80, with the one-second window ending every 0.25 s from +1.0 to +8.0 s. The script refuses to draw unless the pooled counts at +2.0, +3.5, +4.25 and +6.5 s equal the static figure's 55, 82, 99 and 99 of 120. It reads no test recording. Counts: [`decision_clock_counts.csv`](../results/decision_clock/decision_clock_counts.csv); the final frame is saved as a PNG for viewers that do not animate. The same caveats apply: feedback is present in training runs and its onset is not recorded.

| Session | +2.0 s control | +3.5 s early | +4.25 s later | +6.5 s later |
|:--|--:|--:|--:|--:|
| P1 PRE | 8/20 | 17/20 | 17/20 | 12/20 |
| P1 POST | 9/20 | 20/20 | 20/20 | 20/20 |
| P2 PRE | 9/20 | 13/20 | 13/20 | 15/20 |
| P2 POST | 10/20 | 12/20 | 17/20 | 20/20 |
| P3 PRE | 8/20 | 9/20 | 16/20 | 17/20 |
| P3 POST | 11/20 | 11/20 | 16/20 | 15/20 |
| **Total** | **55/120** | **82/120** | **99/120** | **99/120** |

This is **not** a causal effect of waiting longer: the earlier and later models are fitted on different EEG windows; imagery, cue processing, stimulation and artifacts may all change. Results are exploratory because the training tails have been inspected. The six sessions are repeated observations from **three people**.

## Selection and negative controls

At +3.5 s and budget 60, the [chronological training-only policy](early_selection.md) chooses CSP or a Riemannian model inside each 60-trial calibration prefix, refits on that prefix, then scores the shared 20-trial tail: **75/120** versus **82/120** for fixed CSP. P2 PRE is **8/20** for selected versus **13/20** for fixed CSP. This is evidence *against* claiming the selector improved early decisions on these recordings; do not choose a replacement model based on the same exposed tail and call it independently validated.

The [prequential control](../results/prequential_summary.csv) starts at trial 21, trains only on earlier trial labels and refits for each new trial. Fixed post-cue bandpower gets **214/360**, matched pre-instruction power **162/360**. The [mechanism audit](../results/mechanism_audit.csv) uses the *same* later 60 trials per run: post-cue CSP **245/360**, post-cue cue-locked voltage **197/360**, pre-cue CSP **169/360**. The voltage feature ends at +2.54 s; CSP ends at +3.5 s, so these are **not equal-latency decoders**. A separate blocked-fold [cue audit](../results/cue_audit.csv) reaches **69/80** for P1 POST early voltage versus **40/80** before cue. A pre-cue score near 50% does not prove imagery specificity or exclude feedback.

## Timing and streaming

- The original offline [`run`](../results/session_results.csv) pipeline filters the entire run bidirectionally, selects the training-fold model, then evaluates a 2.5–6.5 s window on test: **433/480**. [`compare`](../results/model_comparison.csv) compares a *different*, test-exposed Riemannian pipeline at that window (**449/480**), under equal trial counts. The selection history and distinct feature banks make neither a prospective win. The Riemannian state reset/partition checks are documented in [stream validation](riemann_stream_validation.md).
- The one-pass [`decision-time`](../results/decision_time.csv) curve scores each endpoint with a separate model and causally filtered one-second window; it is **not** one anytime classifier. [`locked_time`](locked_time.md) chooses a session-specific cutoff from blocked training CV (**388/480** test at those cutoffs); the global [training-locked endpoint](training_locked_endpoint.md) uses a different 60-label training-tail rule (at +4.25 s: **407/480** test with all 80 labels refitted). Both test outcomes are retrospective.
- [`organizer_metric`](organizer_metric.md) takes the maximum over time **after** averaging trial correctness on its stated grid. Summing six test-selected session peaks (**427/480**) is not a locked device decision and differs from the global pooled peak (**407/480**). These computations use already-inspected test data.
- A `[2.5,3.5)` s EEG window and 64-sample chunks at 256 Hz can emit the replay decision as late as **+3.746 s** after trigger in the saved recordings. No hardware, display or FES latency was measured. `CausalTrialDecoder`, `CausalSpatialDecoder` and the Riemannian stream wrapper fail closed on processing errors; processed chunks cannot safely be retried. This is **recorded-EEG replay**, not real-time clinical stimulation.

## Channels and transfer

The [montage check](../results/montage_check.csv) finds a stronger signal-topology fit to the bundled `montage.png` layout for P1 POST and the bundled paper's layout for the other files (with a CP5/CP1 swap in P1 PRE). **No MAT file supplies authoritative electrode names or left–right orientation**. The map helps audit within-recording sensor patterns but cannot prove lesion hemisphere or affected hand. Earlier fixed-column C3/C4 outputs should not be used; [mu-power reproduction](../analysis/mu_power/README.md) looks up channels per file and saves untracked outputs to ignored `results/mu_power/`.

[`transfer.csv`](../results/transfer.csv) fits only on the source **training** run and compares 10 named, apparently shared channels to all 16 stored columns. Some target sessions use the later POST session as source, so these are **not** prospective PRE→POST transfer claims. The inferred channel mapping and other recording differences confound comparisons. [`feedback_timing_check.csv`](../results/feedback_timing_check.csv) finds no reliable fixed onset by the tested probes; variable, hand-dependent or test-conditional feedback remains possible. Do not label +3.5 s a measured FES switch.

## Verification and safe updates

```bash
python -m pytest -q
python tools/repo_preflight.py
```

The first command tests algorithms with synthetic data and verifies saved results/receipts; it does not download EEG. The second checks **staged/tracked files in a Git checkout** and refuses raw MAT/RAR, cached artifacts, fitted models and failed notebook outputs. It cannot inspect untracked raw data or rerun data-dependent models. The tracked [CI workflow](../.github/workflows/verify.yml) runs data-free checks. For numerical changes, rerun the corresponding command against SHA-256-verified official data and update *only* the active receipt after execution. Preserve dated historical snapshots. Review `git diff --cached --stat` before committing; keep `data/`, fitted models, caches and personal paths out of Git.
