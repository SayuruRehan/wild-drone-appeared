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
