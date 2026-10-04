# Accuracy and calibration track

Start with [the results report](../../results/accuracy_calibration/REPORT.html), [the fixed protocol](PROTOCOL.md) and [the verification/research handoff](../../results/overnight/MORNING_REPORT.md). This is a separate draft model-track contribution; team acceptance and the organizer's numerical scoring contract remain separate. The physiology notebook is reviewed in PR #1.

Anna's team baseline reproduced **433/480 (90.2%)** exactly in all six sessions, counts 74,74,66,79,70,70. The new common causal-window CSP baseline also scored **433/480**, with different session counts; the training-selected pipeline scored **428/480 (89.2%)**. The same [2.5,6.5) s window and 80-trial denominators are used, but the new models have causal trial-local filtering and different covariance/CSP processing.

Riemannian features gave the strongest nested training **accuracy**, **437/480 (91.0%)**, and an exploratory exposed-test score of **446/480 (92.9%)**, 13 more correct than Anna's reproduced baseline. This is a combined pipeline comparison and a candidate for new independent validation, not a fresh held-out improvement. PRE-training→POST-test scored **40/80, 52/80 and 65/80** for P1–P3. The calibration curves and probability reliability plots retain poor cells and subset variability. The separately reproduced Sudip/nxxis result, 444/479 over [2.5,8) s, is an additional reference, not Anna's baseline.

The fixed causal CSP baseline is better on training probability scores: mean Brier 0.083978 versus 0.098683 and mean log loss 0.297291 versus 0.343823 for Riemannian (lower is better). Its training accuracy is 422/480 versus 437/480. Riemannian improves two sessions, ties one and loses two trials in each of the remaining three. This compares complete pipelines, including bands and classifier, rather than geometry alone.

The +6.5 s endpoint is an EEG cutoff. Delivered Riemannian decision latency and streaming export are unverified. The six verified local streaming models use the separate early continuous-filter CSP pipeline.

## Reproduce

Use a separate scientific environment with Python 3.11 or later for the optional accuracy dependencies. The original study runtime was Python 3.13.14, NumPy 2.4.6, SciPy 1.16.3, scikit-learn 1.8.0 and pyRiemann 0.12, with array-api-compat 1.15.0 and array-api-extra 0.11.4. The overnight receipt records its separate runtime and exact comparator reproduction. These differ from the original full-environment pins; all six legacy accuracies nevertheless match exactly. Installation inside a virtual environment avoids changing other scientific work.

From the repository root:

```sh
python analysis/accuracy_calibration/run.py --data-dir /path/to/stroke-rehab --output results/accuracy_calibration --stage all
python analysis/accuracy_calibration/validate.py
python analysis/accuracy_calibration/build_report.py
python -m pytest -q
```

For an explicit execution boundary, run `--stage train` first, inspect its saved selection receipt, then `--stage evaluate`. `--stage baseline` reproduces Anna's baseline and `--stage plot` redraws saved metrics. The train stage opens training files only. The temporary training covariance cache is ignored by Git and is not a delivery artifact. Recordings remain in the organizer's local data directory.

Install this optional comparison with `python -m pip install -e '.[accuracy,test]'`. The report's accuracy tables are calculated from CSVs; figures are reproducible PNG/SVG outputs. `validation.json` now independently checks the published CSV arithmetic, OOF labels/predictions, calibration memberships and frozen 80-label endpoint without recordings or the ignored covariance cache. It does not run or claim a pytest count; the separate dated `results/overnight/FINAL_CHECKS.json` records test execution. The original 37-test study remains historical evidence. `provenance.json` records input hashes, runtime and the model-selection receipt hash. Scoring/submission checks and the unsent organizer inquiry are saved in the Competitions owner folder; no email or submission was sent. The comparison was moved into an isolated worktree based on Anna's `ffcdb7e` after a concurrent owner-record correction; the independent new-model CSVs were preserved without numerical changes, and her baseline was then reproduced in this worktree.
