# Evidence for the overnight G27 review

Reviewed on 4 October 2026 against the local accuracy/calibration study at `c1a6d42339b191bfc3c23a7d4da64cd8487cfdc8`. These findings support a comparator and evaluation checks; they do not establish rehabilitation benefit or co-adaptation from the retrospective recordings.

## Methods and evaluation

- **Barachant et al. (2012), Multiclass Brain–Computer Interface Classification by Riemannian Geometry.** Covariance geometry and tangent-space representations are established alternatives to CSP. With sixteen channels, each symmetric tangent representation has 136 features, or 408 across three bands. That is large relative to eighty training trials; regularization and fitting each transformation within its training partition matter. Healthy-participant benchmark results do not establish stroke performance. [Author publication record](https://alexandre.barachant.org/papers/journals/IEEE-TBME-2012/), [author manuscript](https://www.researchgate.net/publication/51727880_Multiclass_Brain-Computer_Interface_Classification_by_Riemannian_Geometry).
- **Bleuzé, Mattout and Congedo (2022), Tangent space alignment: Transfer learning for Brain-Computer Interface.** Recentring, rescaling and supervised alignment can support transfer, but results depend on paradigm and dataset. The strongest aggregate improvement was for ERP, and a baseline-performance restriction in principal averaged analyses limits applicability to difficult stroke recordings. A later source-plus-target calibration comparison is justified as a hypothesis, not a guaranteed improvement. [Full primary paper](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2022.1049985/full).
- **Jayaram and Barachant (2018), MOABB: Trustworthy algorithm benchmarking for BCIs.** Rankings vary across datasets; comparable preprocessing and evaluation aligned with the intended generalization question are essential. Matching trials, windows and outer folds here follows that principle. Investigating the current shuffled-inner/blocked-outer distinction is our methodological inference, not a specific prescription from this paper. [Primary manuscript](https://arxiv.org/html/1805.06427).

## Human and decoder learning

- **Rajeswaran et al. (2026), Assistive algorithms influence neural representations in motor brain-computer interfaces.** Published in Nature Communications on 15 September 2026; this supersedes the slide's bioRxiv 2024 citation. Assistive/adaptive decoders can influence learned neural representations. Animal observations and controlled network simulations require separate interpretation, and concentrating predictive information into fewer modes does not automatically mean lower overall neural dimensionality. These mechanisms cannot be established with our offline EEG classifier. [Full primary paper](https://www.nature.com/articles/s41467-026-76109-y).
- **Madduri et al. (2026), Computational framework to predict and shape human–machine interactions in closed-loop, co-adaptive neural interfaces.** Published in Nature Machine Intelligence on 23 March 2026. Fourteen people used a myoelectric interface; adaptation rate affected performance and human adaptation, with slower adaptation outperforming faster adaptation in the tested setting. This is EMG control, not stroke EEG, and does not establish a universal optimal rate. It motivates separate measurements of assisted control and independent user learning. [Full primary paper](https://www.nature.com/articles/s42256-026-01194-z).

## One bounded sensitivity analysis

The current `select()` uses shuffled stratified inner folds while the outer evaluation uses contiguous purged blocks. This is not demonstrated leakage: the outer score evaluates that exact algorithm. However, selection optimizes a different temporal generalization problem. Outer fitting includes trials on both sides of validation blocks, so it measures blocked interpolation, not prospective deployment.

Retain the six predefined candidates, features, seeds, tie rule and five outer folds. Once, replace only inner selection with three contiguous validation folds and a one-trial purge based on original trial indices. Record memberships, class counts, selected settings, and per-session out-of-fold accuracy, Brier score and log loss. Fail explicitly if a fitting set lacks either class. Preserve original results; do not rerun exposed tests, enlarge the search, or assume improvement.

Anna's message to the other collaborator requests their time-resolved accuracy comparison and figure. This sensitivity supports that work and does not duplicate its window sweep.

## Longer-term hypotheses

1. Does source-plus-target alignment reduce calibration burden without negative transfer? Compare source-only, target-only and combined alignment at identical label budgets, fitting alignment on permitted calibration data and evaluating held-out participants/later sessions. Report individual failures.
2. When does assistance improve control while weakening independent learning? Compare fixed, slow, fast and tapered adaptation first in a controlled simulator with matched initializations/seeds. Measure assisted error, unassisted retention and perturbation robustness separately. Prospective human validation would remain necessary.

Current zero-label transfer and target-only labelled calibration are different fitting regimes. Balanced random calibration subsets are retrospective label-budget experiments; they are not chronological acquisition times or a measured learning trajectory.

## Smallest mechanism experiment — proposed after review, 5 October 2026

Use one synthetic two-learner simulation before involving EEG: a user/encoder maps a target into noisy features, and a decoder maps features into an output. Compare fixed, slow and fast decoder updates at declared rates, with the same initial parameters and noise streams for ten paired seeds. Include a frozen-encoder control so improved decoder tracking of drift cannot be mistaken for user learning. Measure assisted task error, encoder change/effort, unassisted retention with the final decoder frozen, and concentration of target information across feature directions. Define the synthetic update rules and each measure before running; a reduction in predictive directions is not automatically reduced overall neural dimensionality.

Output: one runnable notebook and one table, CPU only, maximum one hour of execution after implementation. Do not sweep rates to choose a winner. Start with synthetic features at the covariance/tangent readout interface; the current nonlinear covariance-to-tangent transform need not be simulated to test two-learner dynamics. Connect this later to the existing NeuralBench pipeline through a separately defined perturbation/transfer experiment. This is a mechanism hypothesis motivated by Rajeswaran and Madduri above, not evidence about stroke EEG, an accepted competition adaptation policy or a launched new project. Existing Fabrizi data and findings remain separate.


## What the current methods actually do — 5 October 2026

The forward audit fits a labelled prefix and predicts the same later trials
61–80 at declared EEG cutoffs. It measures retrospective chronological
performance on six training sessions; it does not measure treatment benefit.
The newer selection policy uses only forward folds within that prefix to choose
between fixed CSP/LDA and Riemannian pipelines at +3.5 s. All later tails were
already exposed. At 60 calibration trials, fixed CSP has 82/120 correct and the
selected policy 75/120; the selection policy has not improved the result.

CSP learns spatial projections separating class covariance patterns; LDA then
classifies relative projected band power. The Riemannian method represents each
trial with shrunk covariance matrices, whitens them by a covariance reference,
takes matrix-log tangent features, and uses regularized logistic regression.
Its current-trial reference update uses EEG without labels. Explicit stream
state makes that update persist across covariance batches; it does not itself
filter raw EEG or demonstrate human learning. These two pipelines use different
band banks, so their comparison cannot isolate geometry alone.

The most useful next experiment is fixed recentering on/off at the same early
endpoint and trial IDs, with class recalls and continuous decision-score
ranking alongside accuracy. This separates adaptation effects and threshold
collapse from a change of model family. Freeze it before execution, show P2/P3
individually, and do not select a winning arm from these exposed tails.
Geometry-aware alignment is a later transfer hypothesis requiring confirmed
channel correspondence and identical label budgets. [Rodrigues et al., 2019](https://pubmed.ncbi.nlm.nih.gov/30596565/).

For EEG/NeuralBench, use this as an evaluation extension: matched chronological
splits, class-collapse indicators, held-out subject/session transfer and explicit
adaptation state. For broader NeuroAI, the controlled two-learner simulation
above would test encoder/decoder interaction; offline stroke EEG does not
identify that mechanism. Keep the simulator, EEG benchmark and clinical
interpretations separate.

Affected-hand metadata are unavailable according to the organizer reply relayed
on 5 October. Exploring ERD/ERS asymmetry is reasonable, but clinical side
inference would need independent validation and a confirmed montage. Artifact
and compensatory activity can affect both measured ERD and apparent decoder
performance. [Primary stroke artifact study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6180341/).
