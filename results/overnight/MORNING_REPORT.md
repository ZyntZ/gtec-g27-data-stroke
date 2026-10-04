# G27 verification and research handoff — 4 October 2026

Anna's physiology PR [#1](https://github.com/ZyntZ/gtec-g27-data-stroke/pull/1) is ready for review at `8f44f8d`. It incorporates main `483ac87`; 42 tests and repository preflight pass. The numerical pipeline, notebook execution and model-export checks are complete. The single training-only selection sensitivity is mixed and is not adopted as a universal upgrade.

## Reproduction and source

- Frozen pipeline snapshot `1edae10`, incorporating main `45d5b91`: all nine committed CSVs reproduce exactly across nine CLI stages, including the newer training negative controls. All twelve recording hashes remain unchanged. [Receipt](reproduction.json).
- Notebook snapshot `4285e85`: all three notebooks execute unchanged source from top to bottom (13 code cells, zero errors) in an ignored replica. Training QC and all three physiology CSVs match exactly. All six trusted local model exports reproduce 80 streaming decisions each, agree with batch predictions and timing, and match their training-input hashes. These are execution checks, not model-generalization scores. [Receipt](artifact_checks.json).
- The subsequent main `483ac87` changes README placement and adds local preflight/tests; analysis modules, notebooks, configuration and numerical references are unchanged. Only affected safety checks were rerun. The physiological analysis is also hash-identical on the ready PR branch.
- Reproduction runtime: Python 3.13.14, NumPy 1.26.4, SciPy 1.16.3, scikit-learn 1.8.0. Sensitivity runtime: Python 3.12.13, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, pyRiemann 0.12. The original selected training metrics reproduce in all six runs despite the different environment. The final review branch passes 49 tests in the scientific environment and 46 with one optional module skipped in the default environment; repository preflight and staged-diff checks pass. [Final check receipt](FINAL_CHECKS.json).

## Montage, counts and timing

Direct inspection of the supplied montage confirms C3/C4 at columns **5/9**, correctly implemented as zero-based indices **4/8**. This checks label consistency, not the acquisition reference or individual electrode positions. The physiology calculation retains 80 trials per run, 40 per hand: 480 repeated trials from three people.

Physiology windows relative to held-label onset: baseline **[−1,0)**, early **[2,3.5)**, task **[2,8)** seconds. P1 POST's first training onset is sample 351: its one-second baseline starts at 95 and is valid; a two-second baseline starts at −161 and would be excluded. The reference `mi.py` therefore uses 79 P1 POST trials where this analysis uses 80. Compare identical retained trial IDs, or disclose the denominators explicitly. Per-trial FES timestamps are unavailable, so early post-cue must not be described as definitely feedback-free. The liked reference animation remains unchanged.

## One prespecified selection sensitivity

The complete six-candidate selector originally uses shuffled inner folds inside five blocked, one-neighbour-purged outer folds. We changed only the inner folds to three temporal blocks, purging by original trial IDs even across outer-fold gaps. All transformations remain fold-local; training and validation partitions contain both classes. No test evaluation function is called, and only the six declared training paths are permitted. This is a code-audited I/O policy, not a measured file-open counter.

| Training run | Original shuffled inner | Blocked inner | Difference |
|---|---:|---:|---:|
| P1 PRE | 67/80 | 74/80 | +7 |
| P1 POST | 79/80 | 79/80 | 0 |
| P2 PRE | 71/80 | 64/80 | −7 |
| P2 POST | 71/80 | 71/80 | 0 |
| P3 PRE | 71/80 | 73/80 | +2 |
| P3 POST | 65/80 | 66/80 | +1 |
| Total | 424/480 (88.33%) | 427/480 (88.96%) | +3 |

Mean Brier score changes from 0.09992 to 0.08975; mean log loss from 0.38769 to 0.35043. Individual losses remain in the [summary](selection_sensitivity.csv); [memberships and choices](selection_sensitivity.json) retain all 60 outer records and 12 full-fit choices. These are within-run blocked-interpolation estimates: fitting uses trials before and after validation blocks. The three-person dataset and mixed session effects do not support a general improvement claim. Preserve the original algorithm as the reference; treat blocked inner selection as a reported sensitivity, not an automatically better deployed decoder.

The previous study's Riemannian family scored 437/480 in nested training evaluation and 446/480 on already-exposed tests. Its six-candidate selector scored only 428/480 on those tests. Those historical exploratory results are retained; tonight's sensitivity does not rescore or rehabilitate their validation status. The zero-label source-transfer point and target-only labelled calibration points use different fitting regimes; their connection is not a measured continuous learning trajectory.

## Research implications and next handoff

[Five-primary-paper review](../../analysis/overnight/LITERATURE_REVIEW.md) separates evidence from proposals. The slides' Rajeswaran bioRxiv citation now has a published [Nature Communications version](https://www.nature.com/articles/s41467-026-76109-y). Decoder assistance can influence learned neural representations, while [Madduri's study](https://www.nature.com/articles/s42256-026-01194-z) tests co-adaptation using EMG rather than stroke EEG. Our offline dataset cannot establish those learning mechanisms or rehabilitation benefit.

Anna can review the physiology PR's notebook, methods and configuration changes. Her other collaborator's requested time-resolved comparison should retain identical trial IDs/counts and scoring windows, show early and later periods with explicit feedback uncertainty, and label the endpoint as time after trigger or after cue consistently. Their requested deliverable and team acceptance remain separate from this assistant package. Exact organizer scoring requirements and FES timing remain external unknowns; the organizer inquiry stays unsent.

The model/research review branch is based directly on main `483ac87` and omits the physiology changes in PR #1. Its verification script checks whichever notebooks are available; the three-notebook receipt above was produced in the combined local worktree. At this main snapshot, the README described a GitHub Actions sanity workflow, but no workflow file was tracked. Astra identified this mismatch; the review branch now documents the actual local commands. The runner also now fails on numerical mismatches or missing references while retaining its receipt, and the HTML report distinguishes historical results from current PR publication. Three synthetic runner cases pass; saved EEG-derived numbers are unchanged. Hosted CI is not enabled by this contribution.

Two later directions, not launched tonight:

1. **EEG/NeuralBench:** test whether source-plus-target tangent alignment reduces calibration labels without negative transfer. Compare source-only, target-only and combined models at identical budgets, fitting alignment on permitted calibration data and evaluating held-out people/later sessions. Start with a small CPU benchmark and retain failures.
2. **NeuroAI:** simulate fixed, slow, fast and tapered decoder adaptation with matched initializations. Separate assisted control error, unassisted retention and perturbation robustness. These mechanisms require later prospective human validation. For other electrophysiology work, the transferable lesson is to test measurement/reference assumptions with explicit controls; no Fabrizi data, anatomy or findings are imported here.

AI-produced analysis is not evidence of Jason's personal understanding or accepted team ownership. A decoder here is the fitted rule translating EEG features into a left/right prediction; the mu-power notebook is a descriptive physiology measurement alongside that rule.
