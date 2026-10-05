# Validation and rejected-input state for PR5

This companion change uses PR5's existing `predict_next`, `decision_next` and
`predict_stream` API. Stateless `predict` and `decision_function` still start
from the training reference. It does not introduce another stream interface.

A covariance batch must match the fitted band/channel dimensions and contain
finite, symmetric positive-definite matrices. Invalid input is rejected before
state advances. If the classifier fails, both reference matrices and trial
count return to their state before the call, including a failure partway
through `predict_stream`. A finite nonnegative reference prior is required.
Tests cover invalid input, partial failure and the normal reset/parity behavior.

Model-level rollback does not restore the EEG stream's filters, trial accumulator
or sample position. `CausalRiemannDecoder` therefore fails closed: any processing
exception raises `StreamDecoderFailedError` and makes that decoder terminal.
Further calls are rejected even if the original classifier problem is removed.
Recovery requires constructing a new decoder (which resets model references)
and replaying all EEG/onsets from a known run boundary. Retrying only the failed
chunk or resetting only the model is unsupported. Regression tests inject a
classifier failure in the chunk completing a trial, verify model rollback and
terminal rejection, then verify reconstruction; invalid EEG also fails closed.

`results/forward_calibration/stream_parity.json` records a separate reproduction:
all **3,840** historical training-only predictions and **192** summary rows match.
The original main provenance remains intact; PR5's incoming provenance snapshot
is preserved separately. Source text hashes normalize CRLF to LF; recording
hashes remain raw. No test recording is opened by this reproduction.

For all six training recordings, fits on trials 1–60 give the same tail labels
and final references for one-trial calls and 3/7/10-trial partitions, with
adaptation on and off. Batch and sequential decision scores agree at
`rtol=1e-10, atol=1e-12`; final references agree at `atol=1e-12`.
PR5's raw EEG replay also reproduces all 80 batch labels per training run with
64-sample chunks. This whole-run replay checks implementation equality; it is
not a predictive score because calibration samples are included in the replay.
EEG cutoff is +3.5 s; the chunk containing that cutoff determines emission.
Hardware/transport latency, per-trial FES onset and clinical benefit are unmeasured.

Reproduce locally:

```sh
python analysis/reproduce_stream.py --data-dir /path/to/stroke-rehab
python -m pytest -q
```

PR5 head `d828b6d` is the integration base. Main `dbab3b7` was inspected:
its new cue/prequential controls do not alter these Riemannian features.
This is a companion for PR5 review, not a request to merge two competing APIs.

An isolated combined-code check with PR7 also matched all 24 early choices,
1,920 outer predictions and 1,080 inner predictions; see
`results/forward_calibration/combined_early_parity.json`. After integration,
regenerate PR7's early provenance because its Riemannian dependency hash changes.
The equality check supports unchanged decisions; it does not make either
analysis an independent validation set.
