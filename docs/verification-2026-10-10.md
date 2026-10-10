# Study integrity fixes — 10 October 2026

Addresses issues #16–#21 following PR #15, based on master commit
`653a323128d0024bb76c895c4458a70f01e49d01`.

- Generic checkpoint loading preserves the feasible-action mask and training identity.
  CLI evaluation labels these policies `masked <algorithm>` and rejects unknown masks.
- Completed fit reuse requires matching algorithm, environment, training settings,
  budget, mask, checkpoint architecture and complete training/validation reports.
  Both study runners check all jobs before launching workers. Mismatches fail without
  overwriting old results; use a new output folder for a different experiment.
- Final-study training and validation cannot overlap the reserved final-test scenarios.
- Masked Double DQN selects a feasible action with the online network and evaluates
  it with the target network. Standard masked DQN remains the default. New exports
  record the target rule; legacy masked checkpoints claiming Double DQN without that
  marker are rejected because the earlier implementation ignored the flag.
- Notebook rerun mode reads its new outputs, including learning curves and validation
  references. Archived mode continues to display the historical study. Current-run
  review uses current metrics; the archived interpretation is shown in archived mode.
  The documented sensitivity command explicitly selects DQN.

Validation on CPU: **156 tests passed, 5 CUDA-only tests skipped**; Ruff passed;
all 16 notebook code cells executed in archived mode without errors. Regression tests
also execute rerun table cells using distinguishable fresh result fixtures. The full
300-episode study was not rerun as part of these correctness checks.

The new regressions failed on the original implementation before the corresponding
fixes. Independent review approved the final code after report-schema/finite-value
preflight and legacy Double DQN checks were added.

These are correctness and reproduction fixes, not evidence of better learned
coordination. Historical experiment artifacts have not been changed. Scale-up to the
proposed 40×40 task remains conditional on demonstrated 20×20 improvement.
