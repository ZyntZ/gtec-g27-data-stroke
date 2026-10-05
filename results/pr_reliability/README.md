# Integration checks

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
Local runs report that module as skipped if absent. Numerical reproduction uses six local training MAT files
only. A reproduction of earlier predictions is not independent validation.
