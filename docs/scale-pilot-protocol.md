# Prospective 40×40 supervised inspection and mission pilot

PR #29 showed a full 20×20 mission improvement after broader training coverage.
This newly authorized pilot moves the proven supervised recipe to the assignment
domain. Freeze source/protocol before collection, fitting or rollout. Do not
present this as trained DQN/REINFORCE/A2C, a final study or isolated grid scaling.

Use `configs/main.json` environment, including defaults for unspecified fields:
40×40, horizon 300, battery 160, recharge 8, radius 2, deadline 30, max fuel 180.
The preceding environment was 20×20/160/80/5/1/25/100. These are multiple domain
changes. Keep existing physics, rewards, masks, centered transform and model
class unchanged. Grid-specific flattening enlarges the parameter count; report
it rather than claiming the network has the 20×20 size. Fit from scratch, never
silently reuse a 20×20 checkpoint with incompatible tensors/configuration.

Collect training worlds 0–199 with the same world-seeded masked-random and native
Greedy collectors at steps 30/60/120/240: 1,600 planned states. These snapshots
cover early and late phases of the new 300-step missions. Fresh ranking worlds
6800–6815 use the same collectors/times: 128 planned states. Persist both packets
before any fitting. Log genuine early termination; retain zero-range states in
the archive and exclude them only from fitting and informative-state means.
Full-mission development worlds 7000–7019 are separate from training and ranking.
Confirmation worlds 5800–5819 and final-test worlds 10000+ remain unopened.

Fit three centered networks, seeds 0/1/2, exactly 2,000 Adam updates at .001,
batch 64, gradient clip .5, uniform informative-state sampling with replacement,
one CPU thread and deterministic Torch algorithms. This matches the preceding
budget; do not extend it after inspecting results. Export initial/final models,
all losses, sample-sequence digests/counts/exposure and source/data/model identities.
All three fits finish before evaluation predictions are read. No evaluation
labels enter an input, fit, action-mean control or checkpoint selection.

Measure final own-training/fresh normalized gain regret, tied-best action match
and masked MSE. Retain initial fresh predictions, the training-action-mean
control, exact inspection reference and uniform feasible-random expectations.
Initial training scores are outside scope. Keep complete lossless float32 score
vectors. For network metrics, average informative states within each fitted seed
and report mean/sample SD across three seeds. No significance claim.

Run final models plus action-mean, native Greedy/Sweep and world-seeded masked
random on all 20 mission worlds: seven conditions, 140 episodes. Use existing
first-index masked argmax and `run_mission_validation.rollout`, with full-horizon
zero freshness/full overdue padding for failed fleets. The primary scorer and
controls use the existing emergency-return mask; native Greedy/Sweep retain their
original candidate rule, a disclosed contextual difference. No model labels,
lookahead, RL update or default-policy replacement is involved.

Report all original mission metrics, including return, freshness, overdue,
energy, losses, charging/returns, collisions, steps, intact fleets and failures.
Network mean/sample SD uses three seed means; baseline/control SD uses 20 worlds
and is labeled separately. Energy and charging are descriptive measures.

Keep two pilot decisions separate. Every fitted seed must pass each relevant
condition; do not select the best seed:

1. **Ranking capability:** own-training regret ≤.05, fresh regret ≤.20 and fresh
   tied-best action match ≥25% (same thresholds as the preceding ranking probe).
2. **Mission capability:** freshness ≥90% of native Greedy, overdue ≤Greedy+.05,
   mean losses ≤Greedy+.1, fleet-failure count ≤Greedy, freshness ≥masked-random+.01
   and original return ≥masked-random+.5.

Passing supports a separately designed RL comparison at 40×40; it does not
establish RL success, a deadline guarantee, safety outside the sampled worlds or
assignment readiness. A negative pilot is reported without retuning or running
extra fits. Compare absolute values to native 40×40 baselines; preceding 20×20
numbers are contextual, not a causal grid-size comparison.

Test domain configuration/splits/snapshot bounds, per-seed decisions and a real
miniature end-to-end run with exported score/argmax and checkpoint checks. Keep
prior production files intact; reuse collector, fitter, metric and rollout code.
Verify published source identity before launch. Archive complete ranking inputs,
labels/masks/scores, every mission action/observation hash/reward component,
numeric rows, audits and checksums. Raw dataset/model binaries stay in ignored
`runs/`; record their hashes. Independently recompute rankings and full-horizon
missions without production metric helpers; reconstruct all natural states and
all 125 actual simulator action rewards; replay all full missions and reproduce
every exported score vector. Verify checkpoints remain unchanged. Review frozen
source and final evidence, pass lint/tests/demo and both final CI jobs, merge the
authorized sr→master PR, then synchronize `sr`.
