# Forward-only calibration and decision-time audit

This is a training-run-only retrospective **sensitivity analysis**, not a new
blinded test result or evidence of rehabilitation benefit. It asks whether
conclusions from the existing test-exposed curves survive when fitting uses
only earlier labelled trials and comparison uses the *same later trial IDs*.
For the exact design, input/code SHA-256 hashes and runtime versions, see
[`configs/forward_calibration.json`](../configs/forward_calibration.json) and
[`results/forward_calibration/forward_calibration_provenance.json`](../results/forward_calibration/forward_calibration_provenance.json).

## Run

From the repository root, with Python 3.11+, system libarchive and the
[organizer's downloaded recordings](../README.md#get-running):

```bash
python -m pip install -e '.[test]'
stroke-rehab forward-calibration --data-dir data/stroke-rehab
stroke-rehab plot-forward-calibration
python -m pytest -q
```

To use an external extraction, pass `--data-dir /path/to/stroke-rehab`.
To change the protocol, copy the JSON design and pass `--config PATH`; this
creates a different exploratory experiment and **does not inherit the
committed result table**. The included four-panel plot expects exactly four
decision endpoints; redesign the plot before using another number of stops. `--output-dir` keeps alternative results separate.
No MAT or RAR files belong in a commit. CI runs synthetic tests and preflight
only; the data-dependent command runs locally after an organizer download.

## Evaluation contract

- Six training files, three participants, PRE and POST. Never open test files.
  Each file has 80 trials, 40 per class, and sampling frequency 256 Hz.
- Fit the *first* 10, 20, 40 or 60 trials of the same chronological training
  run, including preprocessing parameters and label-dependent transformations.
  A calibration prefix must contain both classes; early prefixes can be
  markedly imbalanced. The fixed validation set is trials **61–80** for
  *every* budget, method and time point (20 decisions per run). Trials 41–60
  are deliberately unused at smaller calibration budgets, rather than
  borrowing their labels to select a configuration.
- For each trial use a one-second EEG interval **[1,2), [2.5,3.5),
  [3.25,4.25), or [5.5,6.5)** seconds after trigger. The instruction is given
  at +2 seconds. Existing fourth-order Butterworth banks filter continuously
  in one forward pass; spatial projections, shrinkage and classifier are
  fitted on the calibration prefix. Each window uses only samples available
  by its endpoint. There is no measured device-latency guarantee.
- Compare fixed filter-bank common spatial patterns (CSP) + shrinkage linear
  discriminant analysis (LDA) with the existing Riemannian covariance model.
  The two **use different frequency banks** (CSP: 8–12, 12–20, 20–30 Hz;
  Riemannian: seven four-Hz bands spanning 4–32 Hz). The experiment compares
  entire pipelines; it does not isolate the effect of spatial geometry.
  The Riemannian target reference is updated per validation trial from
  its *current and earlier unlabeled* covariance (never a target label).
- Report per-run integer correct/20, mean six-session balanced accuracy, and
  per-trial predictions. Class-balanced accuracy averages the two recalls
  within each run, so a nominal 50% is interpretable even when the last 20
  trials are not class-balanced. Sessions are repeated recordings from only
  three people; the figure's six trajectories do not represent six people.

## Training-only observations

At 60 calibration trials, all methods use the same 120 later training trials
across six runs (not 120 independent people):

| Endpoint after trigger | CSP correct/120 | Riemannian correct/120 | CSP mean balanced accuracy | Riemannian mean balanced accuracy |
|--:|--:|--:|--:|--:|
| 2.0 s (pre-instruction control) | 55 | 47 | 47.0% | 41.3% |
| 3.5 s (early post-cue) | 82 | 75 | 70.2% | 64.7% |
| 4.25 s (later, feedback possible) | 99 | 102 | 83.4% | 87.2% |
| 6.5 s (later, feedback possible) | 99 | 100 | 81.6% | 84.6% |

See the [plot](../results/forward_calibration/forward_calibration.png),
[summary](../results/forward_calibration/forward_calibration_summary.csv), and
[per-trial decisions](../results/forward_calibration/forward_calibration_predictions.csv)
for budgets 10/20/40/60 and every run. The apparent advantage changes with
calibration budget and decision time. In this *training-only, fixed-tail*
protocol the early CSP result exceeds early Riemannian at 60 trials. The
existing 80-training/80-test offline 2.5–6.5 s Riemannian result addresses a
**different** pipeline, window and population of scored trials and was
inspected previously; it cannot be picked over CSP based on this sensitivity.

The rising late signal could reflect motor imagery, movement artifacts,
visual feedback and/or functional electrical stimulation (FES). The data
contain no per-trial feedback markers. In particular [2.5,3.5) s is *not*
proven FES-free, and +4.25 s is not a verified feedback onset. Pre-cue scores
are a negative control but small per-run samples limit their interpretation.
Selection of the best budget/time from the displayed validation tail would
leak those labels into a new evaluation; use another recording to confirm a
chosen policy. No participant-wise generalization, streaming hardware delay,
probability calibration or clinical recovery is inferred here.

## Repository review note

The older offline `stroke-rehab compare` outputs are score comparisons after
visual inspection of the test files, and `stroke-rehab decision-time` fits a
separate decoder for each decision endpoint. Plotting a single curve does not
make it an anytime classifier; a deployable model would need to specify which
state and filters exist when the trial starts. This audit keeps those
limitations visible rather than presenting a single unblinded peak as the
clinical endpoint.
