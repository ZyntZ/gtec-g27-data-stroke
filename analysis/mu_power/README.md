# Training-only mu-power analysis

This complements the classifier with a descriptive measurement of 8–13 Hz
power at C3/C4. It uses all 80 trials in each of six training runs, from
three participants. No classifier is fitted and no test recording is loaded.
Prepared with AI assistance and reviewed for this contribution.

Open [the executed notebook](../../notebooks/03_mu_power.ipynb), or run from
the repository root with Python 3.11 or later:

```text
python -m pip install -e '.[notebook]'
python analysis/mu_power/mu_pass.py --data-dir data/stroke-rehab
```

For data stored elsewhere, give the script `--data-dir`, or set the notebook's
`STROKE_REHAB_DATA` environment variable to the extracted organiser data folder.
Results go to `results/mu_power`. Raw recordings must remain outside Git.

Each trial compares task power with its own pre-attention baseline [−1, 0) s:
`10 * log10(task_power / baseline_power)`. C3/C4 are columns 5/9 in the supplied
montage. Task windows are [2, 8) s and [2, 3.5) s after held-label onset.
Window-local Welch spectra use one-second Hann segments, 50% overlap,
segment mean removal, a one-Hz grid and trapezoidal 8–13 Hz integration.
The original reference is retained; no artifact rejection or continuous
filtering is applied. Physical amplitude units remain unverified.

P1 PRE full-task medians are C3: left +1.34 / right +0.36 dB, and C4:
left +1.40 / right +0.46 dB. Positive means more band power than baseline.
The distributions overlap and other runs differ. This does not establish
clean imagery, cortical localisation or treatment benefit.

The full-task window includes documented feedback. Actual feedback timestamps
are absent, so the earlier window is not proven feedback-free. The baseline,
early and full-task estimates average one, two and eleven segments respectively;
low or noisy baselines can affect the ratio. Repeated trials are not independent
participants.

Input hashes, parameters, runtime versions, synthetic estimator checks and CSV
readback are recorded in `results/mu_power/provenance_and_checks.json`.
