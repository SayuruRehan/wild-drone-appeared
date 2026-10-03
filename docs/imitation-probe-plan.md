# Supervised capability diagnostic on the existing network

The parameter screens locate repeated boundary commands and battery-return failures,
but do not establish whether representation, supervision or sequential state coverage
is the limiting factor. Before changing architecture, test the unchanged FleetNetwork
against explicit greedy demonstrations. This is supervised imitation, not RL, and its
logits are not Q values. Do not add its results to plain-RL fitted-model aggregates.

Fix the budget before examining outcomes: collect greedy trajectories on training
scenarios 0–19 (20 missions) and validation scenarios 5000–5004 (five missions).
Preserve complete normalized observations, current nine-map/24-vector transform,
125 joint-action outputs and the unshielded argmax inference rule. Train three models
with seeds 0/1/2, Adam learning rate 0.001, batch size 64, 100 epochs, gradient norm
clip 0.5 and ordinary joint-action cross-entropy. No critical-state oversampling,
class weighting, architecture change or RL fine-tuning belongs to this experiment.

Evaluate label metrics every ten epochs: mean cross-entropy, exact joint-action
agreement, per-drone command agreement, and critical-return/charging subsets.
Selected checkpoints maximise validation joint-action agreement; preserve final
checkpoints separately. Validate both on autonomous missions in the same five validation
scenarios, alongside untrained networks and the greedy teacher. Teacher-state accuracy
and autonomous policy behavior measure different things. Record executed critical
return moves, losses, freshness, boundaries and collisions using existing diagnostics.

Interpretation is limited: successful fitting would show that this network can learn
some demonstrated decisions under this budget, not prove that the current RL objective
or training distribution can learn them. Poor fitting would not prove a representational
impossibility. Poor autonomous behavior despite label accuracy motivates investigating
state-distribution shift and accumulated errors, without establishing a single cause.
See [Ross, Gordon and Bagnell (2011)](https://proceedings.mlr.press/v15/ross11a.html)
for the sequential-distribution issue; this probe does not implement their DAgger method.

Training and validation scenario IDs must be disjoint. Validation labels never enter
gradient updates. Save supervised checkpoints under an explicitly distinct format,
verify exported tensors and metadata after saving, and record dataset/source hashes.
Reruns use new folders; this short diagnostic has no resume interface. Final test scenarios
10000+ remain unused. Keep plain-RL code/defaults, physical rules and previous records intact.
Test collection, label leakage, inference roundtrip, split/preflight rejection and artifact
identity; request independent review, check CI and merge via `sr` → `master`.
