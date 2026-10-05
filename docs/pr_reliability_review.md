# G27 pull request reliability review

Version 2, 5 October 2026. Code snapshot `558699d`, combining main `0413792`
with PR5, PR7, PR8 and PR9. This replaces the historical pre-update03 report
at `173a3a3`, retained in Git history. The current one-page PDF names its code
snapshot on the page. Rebuilding it checks the recorded calculation/check
sources and organizer outputs; changed sources require verification first.

## Findings and fixes

| Finding | Cause and response |
|:--|:--|
| PR5 advanced model state before classification returned | A classifier exception could change the reference/count without delivering a decision. PR8 commits state after success and restores a failed covariance batch. |
| Original PR8 could not retry consumed EEG safely | Model rollback did not rewind the EEG/filter/sample state. Updated PR8 makes processing failures terminal. Recovery reconstructs the decoder and replays from a known run boundary; retrying the consumed chunk is rejected. |
| Receipts drifted when branches were combined | PR7/8 changed the Riemannian source, and main update03 introduced another receipt mismatch. Actual reruns refresh active evidence and preserve historical snapshots. Guards compare declared current sources, output hashes and prediction correctness. |
| Reports could overstate chronology or currency | Balanced random subsets are retrospective, not chronological calibration. Captions distinguish inspected test reuse and historical receipts. The PDF now uses the current 136-test snapshot and includes the update03 repair. |

## Verification and review

`results/pr_reliability/verification.json.current_stack_checks` records the
actual organizer/stream execution, source fingerprints and final combined
checks. **136 tests**, repository preflight and all **19** published-result
checks pass. [Hosted verification on PR6 snapshot558699d](https://github.com/ZyntZ/gtec-g27-data-stroke/actions/runs/37296588128)
also passes. CI verifies source and saved results without rerunning raw EEG.

The organizer rerun preserves three CSV byte hashes and twelve raw recording
hashes. Training-only forward replay preserves 3,840 predictions, 192 summaries,
twelve parity checks and six 80-trial raw runs. Early outputs remain unchanged.
The organizer calculation uses six training and six already-inspected test
recordings; it must not be described as training-only.

The requested Astra/Ultra source/integration review is recorded as completed;
its actual runtime model identity is not independently exposed. A separate
completed read-only CLI review verified **Claude Fable 5.1**, configured **Max**,
on immutable `558699d`. Claude read saved evidence rather than executing EEG
calculations and found no blocking code or provenance defect. Root checks also
confirmed 34 active fingerprints, saved organizer arithmetic and Git ancestry.
This report correction uses those completed reviews; neither model reviewed
this documentation revision in a new run. Review metadata is summarised in
`results/pr_reliability/report_snapshot.json`; raw Claude output remains local.

Claude's medium finding was the outdated PDF/builder, corrected in version 2.
Its remaining low-severity suggestions are optional hardening: recompute
organizer aggregates within CI, require the full early-source manifest rather
than its current minimum, visibly label older forward-provenance references,
and broaden private-path checks beyond notebooks. One historical stream
receipt hash could not be reconciled by Claude; a line-ending explanation is
unverified. These do not invalidate the checked active receipts, and are not
silently marked fixed here. No further model search or numerical rerun follows
from this documentation correction.

Workflow coverage is limited to PRs containing `.github/workflows/verify.yml`,
pushes to `main` or `codex/pr-reliability`, and enabled merge groups. It does
not imply a hosted check exists for the standalone PR7/8 branches.

## Results and their limits

| Comparison | Correct / total | Meaning |
|:--|:--|:--|
| Organizer-style CSP at fixed +3.5 s | 358/480 | Fixed scoring time |
| Single pooled time-course peak | 407/480 | One time selected after inspecting test performance |
| Sum of six session peaks | 427/480 | Each session's time selected after inspecting test performance |
| Earlier offline team baseline | 433/480 | Different offline scoring contract |
| Earlier Riemannian comparison | 446/480 | Exploratory reuse of inspected test recordings |
| Early training-tail CSP versus selected policy | 82/120 versus 75/120 | Negative selection result; P2 PRE falls from 13/20 to 8/20 |

Different scoring contracts cannot establish an accuracy gain by direct
comparison. Three participants, exposed test/tail recordings, unknown actual
FES timing, unavailable clinical side and unconfirmed acquisition geometry
limit interpretation. The work supports reproducibility and implementation
reliability, not independent accuracy improvement, motor-intent specificity
or rehabilitation benefit. Left/right task labels are not affected/unaffected
hand labels. ERD/ERS-based clinical-side inference would remain exploratory.

## Integration and handover

Anna merged PR6 `558699d` into main at 11:21 UTC on 5 October 2026; the main
merge commit is `88c9ae8`. Its tracked tree is identical to the reviewed PR6
snapshot. GitHub ancestry confirms it includes the exact heads of PR5, PR7,
PR8 and PR9. PR5/7/9 are also marked merged. PR8 remains an open draft even
though its commits are in main; closing it as redundant is optional cleanup,
not another required code merge. PR1/2 are closed replacements/history;
PR3/4 were already merged.

This report correction targets current main directly. The remaining team
deliverable is the video/submission. The side review has not merged a PR itself,
contacted the team or submitted the competition entry.

Sources: [Anna on PR5](https://github.com/ZyntZ/gtec-g27-data-stroke/pull/5#issuecomment-5989971381),
[PR8 failure handling](https://github.com/ZyntZ/gtec-g27-data-stroke/pull/8#issuecomment-5989985404),
[PR7 combined receipt](https://github.com/ZyntZ/gtec-g27-data-stroke/pull/7#issuecomment-5989993987),
[PR6 update03 integration](https://github.com/ZyntZ/gtec-g27-data-stroke/pull/6#issuecomment-5992153702),
[PR9 update03 integration](https://github.com/ZyntZ/gtec-g27-data-stroke/pull/9#issuecomment-5992166008).
