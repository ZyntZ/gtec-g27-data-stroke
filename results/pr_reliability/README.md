# Integration checks

`verification.json.current_stack_checks` records the actual execution of main
`0413792` plus PR6 and PR9. It passes **136 tests with optional accuracy
dependencies**, preflight and all 19 published-result arithmetic checks. All
three organizer metric CSVs and twelve raw input hashes match update03; 3,840
forward predictions/192 summaries, twelve stream parity conditions and six
80-trial replays match. Organizer scoring reads six training and six previously
inspected test recordings; stream replay reads training recordings only.

The top-level fields and hosted/PDF records in `verification.json` describe the
earlier pre-update03 integration. They do not certify this combined source.
The delivered one-page PDF is now version 2, visibly naming code snapshot
`558699d` and the current 136-test result. `report_snapshot.json` records its
evidence scope and both completed model reviews. The builder refuses changed
calculation/check sources or organizer CSVs rather than relabeling stale results.
Anna has merged that code snapshot into main `88c9ae8`; the report correction
is a separate documentation update against main.
`organizer_metric_provenance_update03_snapshot.json` and
`stream_parity_pre_update03_snapshot.json` retain the preceding receipts;
their active counterparts were refreshed by actual execution.

## Historical pre-update03 verification

`verification.json` identifies this integration's executed source, inputs,
unchanged numerical outputs and test/preflight results. It supersedes no
historical receipt: `combined_early_parity.json` records the older stream wrapper, and the `results/overnight/` files identify their own
dated source snapshots. `early_provenance_pr7_snapshot.json` likewise describes
PR7 before the Riemannian implementation changed; `early_provenance.json`
describes a real rerun of the combined tree.

The checked-in HTML embeds PNGs for offline review; standalone PNGs support
previewing and SVGs support vector reuse. They are convenience exports. CSVs,
settings, source/input hashes and reproduction checks supply the numerical
evidence. Random calibration subset captions apply to the PNG, SVG and HTML.

Run these checks on the candidate that will actually be merged:

```sh
python -m pytest -q
python tools/repo_preflight.py
python analysis/accuracy_calibration/validate.py
```

The workflow runs without raw EEG and installs the optional accuracy dependencies.
It applies to PRs containing the workflow, pushes to `main` or
`codex/pr-reliability`, and enabled merge groups. Standalone branches need their
own check evidence. Local runs report the accuracy module skipped if absent.
Historical early/forward reproduction uses six local training MAT files only;
the current organizer rerun also uses six already-inspected test recordings.
A reproduction of earlier predictions is not independent validation.
