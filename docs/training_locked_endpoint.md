# Training-locked decision-time audit (exploratory)

Run from the repository root:

```bash
python -m analysis.training_locked_endpoint --output results/locked_endpoint/summary.json
python -m pytest -q -p no:cacheprovider tests/test_training_locked_endpoint.py
```

The tool reads **only two existing CSVs**, not organizer MAT files. It verifies the same
six sessions, trial identifiers 61–80, binary labels, time grids, and the 80-trial
class counts. Input SHA-256 checksums and caveats are written to the JSON receipt.
No new test EEG is downloaded, no test prediction is fitted, and no classifier is
chosen with test labels in this script.

## Frozen selection rule

Use the one-second causal **filter-bank CSP + shrinkage LDA** (common spatial
patterns and linear discriminant analysis) model. For each recording, fit the
first 60 labelled training trials (as already done in the saved forward-calibration
artifact) and score trials 61–80. Pool the six training tails **at matching
endpoints**. Compare the only shared eligible times: 3.5, 4.25 and 6.5 s.
Choose the highest training-tail total; break ties toward the earlier decision.
Do not choose six different times from the test sessions.

| Endpoint | Training tails (120 trials) | Exposed test (480 trials; model fitted to all 80 training trials) |
|---:|---:|---:|
| 3.50 s | 82/120 | 358/480 (74.6%) |
| **4.25 s, training-locked** | **99/120** | **407/480 (84.8%)** |
| 6.50 s | 99/120 | see the saved receipt/timecourse; not the selected endpoint |

The **4.25 s** lock matches the pooled test peak *by coincidence* in these
exposed CSVs. The six post-hoc patient/session-specific test peaks sum to
**427/480 (89.0%)**, a different, test-label-selected statistic; it must not
be presented as prospective accuracy. The model has been fit with **60**
training labels for endpoint choice but **80** for the stored test curve;
these are intentionally different fits, not a direct holdout from one fit.

| Patient (both sessions) | 3.50 s, test | 4.25 s, test | Delta, correct trials |
|---|---:|---:|---:|
| P1 | 144/160 | 139/160 | −5 |
| P2 | 111/160 | 137/160 | +26 |
| P3 | 103/160 | 131/160 | +28 |

For sensitivity, leaving each **patient** out of endpoint selection yields
6.5 s for P1, and 4.25 s for P2/P3. Looking up these choices on the
*already-inspected* test curves gives 123/160, 137/160, 131/160, respectively
(391/480 pooled). This is not leave-one-patient-out proof of generalization:
all exposed test files had already been used elsewhere in this project.

## What this does not prove

- The published organizer endpoint takes a **maximum over time of the mean
  binary correctness across trials**; the 4.25 s lock is a different,
  deployable-time *selection rule*. The organizer did not specify their exact
  preprocessing/feature model, so these figures do not reproduce their baseline.
- Six sessions come from just **three participants**. Avoid treating 480 EEG
  trials as independent patients or quoting narrow trial-level confidence
  intervals. P1 worsens at the globally selected time.
- Later windows may decode visual/electrical feedback rather than intended
  motor imagery. No per-trial stimulation timestamps or affected-hand labels
  validate a claim of rehabilitation, cortical plasticity or clinical safety.
- Repeated development on these training tails and previously exposed test
  files means **there is no untouched test set here**. Before claiming a
  winning/generalizable decoder, pre-register the model, endpoint, null
  controls and evaluation rule, then assess new people/sessions prospectively.

Use the defensible demo narrative: *a real-time-compatible EEG decision, a
training-derived time lock, subject-level heterogeneity, and explicit safeguards
against feedback and leakage*. Do not sell 89% as prospective performance.
