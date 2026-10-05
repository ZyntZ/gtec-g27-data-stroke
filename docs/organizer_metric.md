# Organizer peak-time scoring: an explicit retrospective replication contract

The organizer has confirmed that their reported score is the **maximum over
time of the mean binary correctness across trials**, after fitting on the
training `.mat` run and evaluating on its paired test `.mat` run. Using the
feedback phase is permitted; all 80 calibration trials may be used; the
hackathon deliverable is a video. This clarification describes the **score
aggregation**, not the preprocessing, exact decoder, sampling grid, feedback
onset, or the fixed windows that produced the published reference percentages.
The time-series decision rule used for the organizer's methods must be checked
against the cited reference material before calling two methods comparable.

For a fixed session, let `c[t,i]` be one if test-trial `i` is correctly
classified at time `t`, zero otherwise. We calculate:

```text
peak_session_accuracy = max_t (sum_i c[t,i] / 80)
```

**Do not** calculate `mean_i(max_t c[t,i])`: this uses a different, impossible
per-trial oracle. **Do not** report `mean_session(peak_session_accuracy)` as the
peak of the pooled six-session curve: each session selects a different time.
When several grid points tie, the earliest is recorded. Grid density affects
the observed maximum; changing it after looking at test scores makes the
result more optimistic.

## Reproduction

From the repository root, after fetching the organizer archive with
`stroke-rehab download`:

```sh
python -m stroke_rehab.organizer_metric \
  --data-dir data/stroke-rehab \
  --output-dir results/organizer_metric \
  --reference-csv results/decision_time.csv
python -m pytest -q
```

The optional `--reference-csv` flag checks the recomputed 138 causal-CSP
counts against the existing `decision_time.csv`; any difference aborts before
writing results. Without that flag, the command still processes the original
MAT files. This command does **not** need the optional pyRiemann dependency.

The **discretionary method** here is the existing three-band common spatial
patterns (CSP) plus shrinkage linear discriminant analysis pipeline. It fits
**one separate decoder at each endpoint** on the *full* 80-trial training run,
filters both runs causally and continuously, extracts a one-second window
ending at that endpoint, and predicts the same 80 test trials. Endpoints are
2.50–8.00 s after trigger in 0.25-s increments (23 points). The first
[1.50,2.50) s interval straddles the instruction at +2 s; later intervals
may include visual and electrical feedback. These choices reproduce the
*repository's causal CSP time sweep*, **not** the numerical pipelines in the
organizer's overview PDF. No test label enters fitting, but the chosen peak
time is selected from test labels after predictions. The +3.50 s window is
shown as an **illustrative early fixed-time comparator**, already known from
prior analyses, not a new blind or preregistered endpoint.

The command writes per-time counts and class recalls, a session peak/fixed
summary, an aggregate summary, a six-panel publication-resolution plot, and a
receipt of input and source SHA-256 hashes. The source dataset, per-trial
labels, filtered epochs, and fitted models are never exported. Hashes provide
reproduction provenance; they do not make reused test data independent.

## Observations on the already-exposed tests

| Run | Fixed +3.50 s correct/80 | Post-hoc peak correct/80 | Peak time (s) |
|:--|--:|--:|--:|
| P1 PRE | 71 | 75 | 3.75 |
| P1 POST | 73 | 75 | 5.25 |
| P2 PRE | 52 | 64 | 4.25 |
| P2 POST | 59 | 79 | 6.75 |
| P3 PRE | 47 | 66 | 4.25 |
| P3 POST | 56 | 68 | 5.50 |

Sum of six per-session test-selected peaks: **427/480 (89.0%)**. A *single*
peak of the pooled six-session curve: **407/480 (84.8%)** at +4.25 s.
Fixed +3.50 s across all sessions: **358/480 (74.6%)**. These three values
have different selection contracts. The original 433/480 figure uses an
offline, zero-phase, multi-second decoder; it is not a matched comparator.
The organizer's published percentages were produced with different methods
and temporal scoring details; do **not** claim they have been reproduced.

The post-hoc peak may capture effects of the cue, visual feedback, muscle
activity or functional electrical stimulation (FES), rather than motor intent.
Its timestamp cannot be selected for a new test patient based on that
patient's unknown labels, and the same recordings have been inspected during
previous development. It is not a prospective online or clinical score.
Even the causal model's window endpoint does not include compute, display,
transmission, or stimulation latency. With only three participants and no
per-trial FES timestamps, no gain over competing teams or rehabilitation
benefit can be established by these results. For a video, show the
post-hoc peak **and** the fixed-time curve with these qualifications.

## Combined-source execution

After integrating main `0413792`, the early/streaming changes and PR9, the
organizer calculation was actually rerun with
`python -m stroke_rehab.organizer_metric --reference-csv results/decision_time.csv`.
All three numerical CSVs remain byte-identical to update03, and all twelve raw
recording hashes match. The active receipt records the executed source; the
original upstream receipt is retained as
`organizer_metric_provenance_update03_snapshot.json` and describes that earlier
source only. Source fingerprints normalize CRLF to LF; input and output hashes
retain raw bytes, with numerical CSVs protected from Git text conversion.
The guard requires all six calculation sources and all four output files.
Exact execution and full combined checks are indexed by the current integration
record; regeneration is not a new method, score improvement or blind evaluation.
