# Matched navigation and energy representation diagnostic

The approved next correction tests explicit drone-to-base and return-energy inputs before
expanding data or compute. The preceding fixed-corpus budget study improved training fitting
but lost the entire fleet in all final validation and post-hoc familiar-world flights.
Fix this protocol before inspecting outcomes; it is supervised imitation, not an RL result.

Collect the same complete real Greedy corpus on training worlds 0–99 and teacher validation
worlds 5000–5004, with unchanged 20×20 development physics and 160-step horizon. Both arms
receive all training labels from their first update. Keep all real pre-action states,
including teacher failures; do not invent post-termination labels or restore batteries.

Compare `padded-control` versus `relative-energy`. Both use the original nine map channels,
24-vector inputs, CNN, 128 hidden units and 125 joint-action outputs. Expand the vector part
of the encoder by 12 inputs in both arms. The control receives twelve zeros. The treatment
receives per-drone row and column offsets from base divided by grid_size−1, Manhattan distance
divided by 2*(grid_size−1), and battery minus necessary Manhattan return cost divided by
battery capacity. Derive these in the network forward path from existing observations only,
for training and deployment alike. Keep negative margins, and zero the added features for
lost drones. The bound excludes hovering and blocked-move costs and is not a safety guarantee.

Matching parameter counts does not make these models identical to the preceding 24-input
network. The new control is necessary; historical results are context, not extra replicates.
Initialize both arms with exactly identical tensors for each seed 0/1/2. Use identical
independent sampler seeds, Adam rate 0.001, batch size 64 with replacement and gradient clip
0.5. Continue optimizer/sampler state for twenty rounds of 1,000 updates: six fits. No
weighting, new learner-state collection, action factorization, masks, shields, reward change,
RL warm start, model selection by autonomous return, or physics change belongs here.

Measure teacher-state decisions at initialization and every round; retain round 0–20, final
and selected exports with exact tensor/metadata, model, byte, data and source identities.
The separate selected export maximizes validation joint agreement over rounds 1–20, earliest
ties and initialization ineligible. This reused-validation selection is biased and does not
measure mission success. Compare autonomous 5k/10k/15k/20k checkpoints and selected exports on
the five validation worlds, plus one shared five-world Greedy baseline: 155 missions.
Budget checkpoints are nested within each fit, not independent fitted models.

Primary evidence is the paired final-arm difference in mission return, full-fleet losses,
freshness, safe returns, completed charging and actually executed critical return moves.
Teacher-state accuracy alone, survival without inspection or a single lucky checkpoint does
not establish useful coordination. Average scenarios within fit before computing mean and
sample SD over three fits; pair arm differences by seed. Baseline SD is across scenarios.
These repeatedly used development validation worlds remain exploratory; final worlds 10000+
are untouched. Do not select a budget retrospectively and claim final generalization.

Use fresh folders without resume; keep earlier records intact. Verify initial tensor identity
and persisted mission/baseline counts, archive raw records and collection CSV bytes, replay
missions independently, and pass review/CI before merging sr→master. If autonomous behavior
remains poor, improve useful training experience before scaling to the full 40×40 problem.
