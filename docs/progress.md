# Starting-release record

Completed: approved design, seeded Gymnasium simulator, four conventional baselines,
independently implemented DQN/REINFORCE/A2C, CLI, full-state CNN, checkpoints, held-out
evaluation, provenance metadata, plots, GIF renderer, notebook walkthrough and CI configuration.

Implementation decisions:

- Use a single-controller Gymnasium environment, consistent with the approved fleet MDP.
  PettingZoo would introduce an interface for separately acting policies that is not required
  by this design. Decentralised policies remain future work.
- Charge the remaining overdue mission cost on total fleet loss, and mark the unserved
  remainder as zero freshness/full overdue in metrics. This prevents artificially favourable
  early failure; the failure penalty is a modelling choice to include in sensitivity analysis.
- Keep checkpoint evaluation on exactly the saved environment configuration. Cross-domain
  interventions require a future explicit evaluation interface to prevent accidental setting
  mismatches; current experiments can train/evaluate separate configurations consistently.
- Train on CPU in this starting implementation. GPU/vectorised throughput is future work
  after a learning pilot, rather than an assumed speedup.

Independent whole-project code review found two important issues, now covered by regressions:

1. Short mission horizons clipped inspection ages below their deadline, incorrectly counting
   unseen cells as fresh. Initial ages now equal the deadline; the cap and normalisation use
   `max(horizon, deadline + 1)`. The regression failed with initial freshness 0.8 and now passes.
2. Monte Carlo policy gradients omitted absolute-time gamma weights and divided by a
   policy-dependent realised episode length. REINFORCE now uses gamma^t and a fixed mission
   horizon; A2C also uses absolute-time weighting. The analytic regression verifies discount
   weights, later rollout offsets and fixed scaling.

Review scope decisions: convergence, hyperparameter adequacy, statistical superiority,
calibrated wildfire physics, novelty and a complete literature audit are subsequent research
tasks. This release offers implementation and smoke evidence, and makes none of those claims.
The risk of treating smoke scores as research findings is addressed explicitly in README.
No additional code-review minors were deferred.

Verification on the final reviewed code: 33 tests passed, Ruff passed, and the paired 8x8
comparison completed all three methods with two training seeds and two held-out scenarios
(20 comparison rows). Separate 20x20 two-episode training checks completed for DQN,
REINFORCE and A2C. The 40x40 300-step greedy demo completed with zero losses, zero collision
attempts, three safe returns and three completed charges. These are smoke/integration checks,
not learning or superiority claims.

Validation commands and final results are recorded in the pull request. Repeat them after
changes; do not reuse the original smoke CSV as evidence for a modified simulator.

# Phase 2: training runtime and validation pilot

Foundation PR #1 was reviewed, passed CI and merged. Version 2 now supports CPU/CUDA/auto,
atomic episode-boundary state with Adam/target/replay/RNG restoration, strict resume checks,
validation checkpoint selection, diagnostic CSVs, learning curves, ablatable spatial location
maps and a Monte Carlo baseline for REINFORCE. A private-repo-compatible Colab notebook
includes a bounded CPU/GPU benchmark, Drive persistence, automatic resume, optional 40x40
training and a downloadable archive. It does not consume final test scenarios during tuning.

Original and revised 80-episode pilots and a revised 200-episode study across three independent
training seeds are recorded in [the pilot report](learning-pilot.md). None outperformed greedy
on validation. Best-checkpoint scores are selection-biased; convergence, final generalisation,
GPU throughput and research novelty are not demonstrated. CUDA is unavailable locally and
must be verified by the notebook. Remaining research includes validation-only tuning/ablations,
a frozen repeated held-out study and simulator sensitivity/calibration.

Independent phase-2 review found three important failure paths, now covered by regressions:

- Inherited Git credential helpers could persist a hidden token. Clone now clears helpers; a
  local authenticated Git fixture confirms fake credentials are not stored.
- An interrupted first export could leave a policy without resume state. Authoritative state
  now commits before regenerable exports; interruption at every export restarts/resumes exactly.
- Unavailable CUDA could overwrite study records before rejection. The runner now checks device
  availability before writing. CLI comparison also preflights every requested resume, and the
  notebook consistently benchmarks/selects CPU when CUDA is unavailable.
