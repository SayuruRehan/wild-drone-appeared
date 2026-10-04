# Fixed supervised coverage and learner-state labeling comparison

The previous unchanged-network probe fits 96.4% of training joint decisions but only
9.6% on held-out teacher states, and loses all drones during autonomous validation.
Test whether broader scenarios or labels on learner-generated states help before
changing architecture or using these weights to initialize RL.

| Arm | Initial trajectories | Added trajectories before rounds 2–5 |
|---|---|---|
| fixed | Greedy scenarios 0–19 | None |
| teacher | Same initial data | Greedy executes on 20–39, 40–59, 60–79, 80–99 |
| dagger | Same initial data | Previous round's unshielded learner executes on those same batches; Greedy labels each visited pre-action state |

Each arm has three fits with seeds 0/1/2, five rounds and exactly 1,000 Adam updates
per round (5,000 total). Use learning rate 0.001, batch size 64, gradient norm clip 0.5
and ordinary joint cross-entropy, sampling uniformly with replacement. Keep optimizer
state across rounds and retain all collected labels, with no weighting/oversampling.
All arms should have identical initial and round-one model tensors within each seed.
This is a new matched control; its with-replacement sampling differs from the previous
probe's shuffled full epochs, so previous fits are not extra independent replicates.

Broader arms use the same 100 training scenario IDs, but not the same state/label counts:
learner missions may terminate early. Record every collected teacher label, executed
command, live/critical/unreachable drone count, and all-stay label counts. Do not interpret
matched gradient updates as matched oracle queries or matched information. Greedy has
explicit proposal rules and can fall back to all-stay when its feasible set is empty;
it is a conventional teacher, not an optimal recovery oracle. No labels after termination,
rescue actions, synthetic restored batteries, action masks, rewards or RL updates are added.

Keep the development environment, full observation transform, FleetNetwork and 125-output
unshielded argmax inference unchanged. The dataset aggregation arm is a bounded DAgger
variant: initialize with teacher data, then use only learner actions during added collection
(beta zero after initialization), continuing the supervised optimizer rather than solving
an exact no-regret subproblem. No theoretical performance guarantee is claimed. The primary
reference is [Ross, Gordon and Bagnell (2011)](https://proceedings.mlr.press/v15/ross11a.html).

Use Greedy teacher labels on validation scenarios 5000–5004 for metrics after each round
and at initialization; never train/collect learner trajectories on validation IDs. Select
maximum validation joint agreement across rounds 1–5 with earliest ties; save the final
separately. Evaluate unshielded autonomous missions after every round and for selected
checkpoints on those same five scenarios, with Greedy as the shared baseline (275 missions).
Selection and repeated use of these scenarios makes this exploratory validation, not a
final generalization test. Final test IDs 10000+ remain unused.

Export an explicitly separate supervised coverage checkpoint family. Audit tensors/metadata
and bind exact checkpoint/data bytes, semantic data/model digests, collection-policy identity,
source hashes and rounds/updates before evaluating. Preserve per-round datasets and audit
CSV, numeric metrics and mission records. Use fresh output directories; no resume interface.
Report final and selected results separately, averaging missions within each fit before
computing sample SD across fits. Report teacher-state critical-command metrics separately
from executed critical return moves during autonomous missions. Preserve earlier records.
