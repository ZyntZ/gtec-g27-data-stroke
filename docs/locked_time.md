# Lock the decision time on calibration data, not on test outcomes

## Reproduce

```bash
python -m pip install -e '.[test]'
stroke-rehab download
python -m stroke_rehab.locked_time
python -m pytest -q
```

Only `stroke_rehab/locked_time.py`, the six paired recording files and existing project modules are needed for the new run. CSV files and SHA-256 provenance are written to `results/locked_time/`; the original recordings remain in ignored `data/`. Check `provenance.json` for the exact recording and source fingerprints. Python 3.11+ and the system `libarchive` library are required for `stroke-rehab download`.

## Question and decision contract

**Can a session-specific time chosen without the test labels recover the large post-hoc peak-of-mean test score?** We fix the previously used causal three-band filter-bank common spatial patterns (CSP) plus shrinkage linear discriminant analysis (LDA) classifier. For each session, we perform five contiguous training-run validation blocks (16 trials each) and exclude one adjacent trial on either side of the validation block from model fitting. Preprocessing is continuous and causal, and *the complete classifier*, including CSP filters and standardization, is refit on each fold. There is one prediction per calibration trial at each predeclared 1-second window endpoint from +3.25 to +8.00 s in 0.25-s steps. We choose the endpoint with most out-of-fold correct trials, breaking ties in favor of earlier decisions. We separately select between +3.25 and +3.50 s and report the fixed +3.50-s comparator. Each model is then refit on all 80 calibration trials and evaluated at its locked endpoint on 80 trials from the corresponding test recording. No test score is used to choose the endpoint.

The candidate grid starts after the +2-s instruction; +3.25/+3.50 s **must not be called feedback-free**, since actual visual/electrical feedback onset is unavailable. Late candidate times may capture feedback rather than motor imagery. Blocked folds avoid random shuffling of time-adjacent trials but include *future calibration trials* in some fits: this is not a forward-only simulation or a nested estimate of the selection rule. The highest calibration out-of-fold accuracy is selection-biased; do not present it as an independent performance estimate.

## Results (previously inspected organizer test files; exploratory)

| Session | Training-selected full time (s) | Full correct/80 | Early selected correct/80 | Fixed +3.50 s correct/80 | Test-label-selected peak correct/80 |
|:--|--:|--:|--:|--:|--:|
| P1 PRE | 5.75 | 65 | 71 | 71 | 75 |
| P1 POST | 3.75 | 74 | 73 | 73 | 75 |
| P2 PRE | 6.75 | 61 | 52 | 52 | 64 |
| P2 POST | 6.50 | 75 | 59 | 59 | 79 |
| P3 PRE | 6.25 | 64 | 47 | 47 | 66 |
| P3 POST | 7.25 | 49 | 45 | 56 | 68 |
| **Total** | — | **388/480 (80.8%)** | **347/480 (72.3%)** | **358/480 (74.6%)** | **427/480 (89.0%)** |

The peak column is the **sum of six retrospectively test-selected session peaks**, taken from `results/organizer_metric/organizer_metric_summary.csv`; it is not the pooled peak (407/480), and is **not an achievable locked-in advance score**. All 18 new locked test counts were checked against the independently generated per-time `results/organizer_metric/organizer_metric_timecourse.csv`. P3 POST is a consequential failure: calibration picked +7.25 s but only 49/80 test trials were correct, below the already-inspected fixed +3.50-s comparator (56/80). The P2 POST gain with a late decision is neither uniformly transferable nor proof of intended motor activity. Our zero-phase, four-second baseline (433/480 in the separate accuracy study) has a different latency and filtering contract; it is **not** a fair online comparator.

**Interpretation and limitations.** The organizer has confirmed that published reference numbers are the maximum accuracy of a trial-averaged 0/1 time course on the unseen `.mat` test files. Their scoring includes feedback. The published CSP/TVLDA curves are not reproduced here: our model fits a distinct classifier at each endpoint. The test recordings have already been repeatedly inspected during development, including the choice of this candidate grid and the fixed +3.50-s comparator. These results therefore cannot be treated as new prospective validation despite the per-session code enforcing a training-only selector. Six runs come from only **three participants**; 480 trials are not 480 independent patients. Without trial-level feedback times, clinical outcomes, or a new blind cohort, no claim about neuroplasticity, therapy efficacy, or improvement over the organizer's published methods is warranted.

**Next precommitted experiment.** Fix one causal pipeline and a decision deadline; choose any session adaptation rule on calibration only, then evaluate once on genuinely undisclosed subjects or newly recorded sessions. Record instruction, visual and functional electrical stimulation (FES) timing; test modality-/artifact-related negative controls and motor-area spatial physiology before claiming to decode motor intention. Report per-subject latency, abstention/feedback rate, balanced accuracy, calibration budget and uncertainty. The hackathon video should juxtapose the *post-hoc* peak with the *locked* 388/480 score and explicitly label the latter exploratory.
