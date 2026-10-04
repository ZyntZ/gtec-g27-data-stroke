# Next plays

The most tempting result is not necessarily the most useful one. The first move is to make the model credible under the actual feedback constraints.

1. **Pin down the scoring contract.** Ask organizers whether they want offline trial classification, stepwise feedback decisions, cross-session transfer, or intent decoding before feedback starts. Confirm whether trigger-aligned FES timing is available. The answer changes the target and the legal features.
2. **Make a feedback-free benchmark.** If feedback timestamps can be supplied, score only windows that end before the first stimulation frame; train a causal filter and use the same allowed latency at inference. Until then, publish early and late scores separately rather than combining them into a marketing number.
3. **Use training-only search.** Test predeclared band edges, shrinkage, spatial-filter counts, and one-second post-cue windows via nested trial-level CV *inside* each training run. Keep the test files untouched by model selection; the six current held-out scores are already exposed to the team.
4. **Study domain shift.** Plot per-channel/band artifacts and pre/post distribution shifts. Evaluate cross-session transfer strictly with training runs first. If calibration time matters, plot training-only learning curves with honest repeated partitions; never splice together adjacent samples from one trial across folds.
5. **Build an online path.** Swap zero-phase filtering for stateful causal processing; constrain decisions to available samples; time the pipeline on a CPU; measure latency, reliability, and false-positive feedback before any deployment claim.
6. **Collect independent validation.** Three patients and six sessions are not a clinical or leaderboard-scale sample. Request more untouched subjects/runs; pre-register the selected endpoint before opening their test labels.

Outputs worth shipping next: a scoring-spec note, a leakage check with synthetic streams, a feedback-timestamp audit, a causal decoder, and an independently evaluated model card. No claim of clinical benefit follows from these recordings alone.
