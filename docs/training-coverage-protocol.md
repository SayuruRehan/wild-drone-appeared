# Prospective 20×20 training-coverage comparison

Test whether broader natural training coverage improves the fixed drone-centered
inspection scorer. This is a bounded extension of the preceding supervised
comparison, approved in the continuing project. Change training corpus size only;
keep representation, network, objective, collectors, snapshot times and fitting
budget fixed. No RL/default/mission changes and no 40×40 run are included.

## Fixed data and fitting recipe

- Environment: exact preceding 20×20 configuration, horizon 160, battery 80,
  recharge 5, sensor radius 1, inspection deadline 25, max fuel 100; all remaining
  fields match the archived parent launch configuration.
- `small`: archived worlds 0–39, 319 observed states, 303 informative states.
- `expanded`: retain every original tensor/record unchanged and append worlds
  40–199. Combined worlds 0–199 plan 1,600 natural states. No generated intervention
  maps, new collector, changed sampling weights or filtered favorable worlds.
- Collect both world-seeded masked-random and Greedy trajectories independently,
  at elapsed steps 16/32/64/96, exactly as before. Log genuine early termination
  and exclude zero feasible-gain-range states from fitting and informative means.
- Primary fresh-development evaluation: worlds 6400–6415, 128 planned states,
  same collectors and snapshot times. Collect and persist before fitting.
- Reused development evaluation: prior worlds 6200–6215, 128 observed states,
  125 informative. These previously examined worlds are secondary, not fresh tests.
- Training and both evaluation sets must be disjoint. Confirmation worlds
  5800–5819 and final-test worlds 10000+ remain unopened.
- Use existing CenteredInspectionNetwork with `arm='centered'`, nine spatial
  channels, 24 scalar inputs, 125 output scores and 127,677 parameters. Keep the
  observation-only transform exactly unchanged.
- For each corpus and seed 0/1/2, fit from scratch for exactly 2,000 Adam updates
  at .001, batch 64, clip .5, uniform sampling with replacement from that corpus's
  informative states, CPU with one Torch thread, deterministic algorithms.
- Initial tensors must be identical across corpora within each seed. The sampler
  starts from the same seed+1000, but sampled IDs/digests differ with corpus size;
  do not claim identical batches across differently sized corpora. Record distinct
  sampled-state counts and exposure (128,000 draws divided by informative states).
- All six fits finish before evaluation predictions are read. Export only initial
  and final checkpoints. Each small-corpus final tensor and all 6,000 small-fit
  losses must exactly reproduce the preceding centered fits; verify parent binary
  identities and source/configuration before launch.

## Measures and prospective decisions

Primary metrics: normalized one-step inspection gain regret (lower better) and
tied-best action match (higher better), under each natural state's feasible mask.
Retain all tied best actions and deterministic first-index score argmax. Average
informative states within fit, then report mean/sample SD across three seeds and
paired expanded-minus-small differences. No significance claim; states and worlds
are not independent fitted models.

Report own-corpus training metrics separately: their means use different training
populations. Also retain regression MSE, initial fresh/reused predictions, each
corpus's training-action-mean control, exact inspection reference and analytical
uniform feasible-random expectations. Only targets used by training/control
construction come from training; no evaluation label enters a model input or fit.

Keep two decisions separate and list every condition per seed:

1. **Data effect:** expanded fresh regret at least 0.05 below matched small regret,
   and expanded fresh tied-best match no lower than matched small match. Every
   seed must pass both conditions to mark a consistent useful data effect.
2. **Capability:** expanded own-training regret ≤0.05, fresh regret ≤0.20 and fresh
   tied-best match ≥25%. Every seed must pass all three conditions.

A data effect without capability supports further diagnosis, not RL or 40×40
promotion. Capability and data effect together support considering a subsequent
RL validation; they still do not establish long-term coordination or mission
performance. A negative outcome at fixed 2,000 updates does not prove larger
datasets cannot help at another budget. Do not extend fitting or change the recipe
after observing the primary comparison.

## Implementation and verification

Extend the existing runner flow in `scripts/run_training_coverage.py`; reuse the
unchanged collector, fitter and metric helpers. Test nested/disjoint world rules,
unchanged parent prefixes, duplicate rejection, per-seed decisions and a small
end-to-end run. Freeze source/protocol on `sr` before collecting or fitting, open
the PR to `master`, and run the full repository tests/lint/demo.

Archive lossless complete float32 score vectors for every measured network state,
all losses and numeric rows, input/gain/mask records, source/dataset/checkpoint
identities, matching/exposure audits and separate numerical/replay verification.
Identical per-seed initial models share one archived initial score file for fresh
and reused evaluation; do not duplicate those tensors. Initial training scores
and unmeasured small-model scores on appended training states are outside scope.
Preserve raw dataset/model binaries locally under ignored `runs/` and archive
their hashes. Compression must not round scores or labels.

Independently recompute every reported row/summary/criterion without production
metric helpers. Replay newly collected natural observations and all 125 actual
simulator action gains per new state; verify mask/target normalization, unchanged
transform, checkpoint metadata/tensors and complete score-vector reproduction.
Verify source and archive identities, review the final revision, pass both final
CI jobs, merge the authorized PR, and synchronize `sr` to `master`.
