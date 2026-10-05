# Training-only C3/C4 mu power

This analysis measures 8–13 Hz power in the six **training** runs (three people,
PRE and POST). It does not fit a decoder or open test runs. C3 and C4 are
looked up per file: only P1 POST follows the supplied `montage.png` (EEG
columns 5 and 9, one-based); the other five training runs follow the bundled
paper's layout (columns 7 and 11). The mapping comes from `stroke-rehab
montage-check` (`results/montage_check.csv`), which infers it from
inter-channel correlations; it is not confirmed by the organizer. Figures and
CSVs made before this change used columns 5 and 9 everywhere and are wrong
for those five runs.

Those names are an explicit supplied-montage assumption, not verified acquisition
metadata for every recording. The new [montage PR #3](https://github.com/ZyntZ/gtec-g27-data-stroke/pull/3)
investigates an alternative layout using signal correlations. That can identify
a hypothesis to check, but not establish electrode identity or left/right
orientation. Retain the calculation on columns 5/9 pending authoritative mapping;
do not use these plots as verified hemisphere measurements.

From the repository root, after `stroke-rehab download`:

```bash
python -m pip install -e '.[test,notebook]'
python analysis/mu_power/mu_pass.py --data-dir data/stroke-rehab
pytest -q
```

For recordings outside the checkout, use `--data-dir /path/to/stroke-rehab`.
The notebook `notebooks/03_mu_power.ipynb` accepts `STROKE_REHAB_DATA` and
writes regenerated CSVs, provenance and figures to ignored `results/mu_power/`.
Notebook outputs are deliberately uncommitted. The notebook may also be run
from the repository root; never commit the original MAT files or the RAR.

For each of 80 trials per run and each channel, compute power during
[2, 3.5) s (early) or [2, 8) s (full task) relative to that trial's
[−1, 0) s pre-trigger baseline. Windows are in seconds from trigger onset;
the hand instruction occurs at +2 s. The statistic is
`10 * log10(task_power / baseline_power)` dB. Window-local Welch spectra use
one-second Hann segments at 50% overlap, mean detrending, 1-Hz resolution,
and trapezoidal integration at 8–13 Hz. The baseline, early, and full-task
windows average 1, 2, and 11 segments respectively; differing precision and
noisy baselines complicate comparisons. No spatial rereferencing, artifact
rejection, or continuous zero-phase filtering is used. Power is in squared
original recording units; the physical amplitude units are not verified.

The `training_mu_summary.csv` export lists per-run/per-hand medians and
quartiles, while `all_training_trial_mu.csv` retains all 960 channel-trial
measurements. Script checks all six input hashes and synthetic 10-Hz power
relationships; it checks CSV readback and records its parameters and package
versions in `provenance_and_checks.json`. The inference unit is the person,
not one of the repeated trials. The full-task window includes feedback;
feedback timing is not available per trial, so **even the early window is
not proven feedback-free**. These descriptive differences do not show
motor-cortex laterality, decoding gains, or rehabilitation benefit.
