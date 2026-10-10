# Prospective 20×20 position/urgency ranking diagnosis

This diagnoses the completed 100-episode, training-seed-0 corrected masked Double
DQN checkpoints from the coordination pilot (shaping weights 0 and 10). No policy
is fitted, selected, changed or deployed as part of this diagnostic. The complete
20×20 development environment is unchanged; no 40×40 experiment is authorized by
a diagnostic outcome alone. Binary checkpoint SHA-256, byte size, environment,
completed budget and exact training configuration must match the frozen pilot.

Collect 32 planned snapshots: fresh development worlds 5900–5907, two independent
collectors (masked random seeded by world; Greedy), at elapsed steps 16 and 80.
Neither learned model collects the states. Log missing snapshots and eligibility
failures. These 32 states are not independent training seeds. Confirmation worlds
5800–5819 and earlier final-test worlds are not opened.

Read all 125 learned Q values. Controls: an untrained identical network (seed 101),
one fixed state-blind Gaussian score vector (seed 202), and a one-step original
freshness-gain ranker. This last reference is an inspection input-response check,
not a long-term optimal-value oracle. Its gain uses exact isolated one-step drone
motion/energy/charging/coverage from environment clones, current priority weights
and the next inspection-age increment. Overlapping coverage is counted once.
Neither the original state nor its RNG is mutated by hypothetical actions.

For every natural snapshot, record feasible-action selection and normalized gain
regret `(best_gain-selected_gain)/(best_gain-worst_gain)`. Report tied reference
best-set size and uniform-random expected regret/best probability. Zero-gain-range
states are uninformative for normalized regret and are excluded from alignment
means; show the denominator. Natural states measure immediate inspection alignment,
which can disagree with a policy that values future coverage, energy or charging.

Construct pairs without consulting learned or untrained scores:

- **Urgency stress test:** keep position, battery, fire, fuel, priority, wind,
  charger and time fixed. Use a fresh age background, with one stale region
  exclusive to each of two feasible proposals. Reverse which region is stale.
  Balance the two regions within each exact priority-weight stratum using the
  smaller available cell count and ascending flat cell IDs. Both variants have
  identical age histograms and priority-weighted global freshness, isolating
  urgency location from urgency amount.
  Search at most 64 proposal pairs by descending product of exclusive weighted
  coverage; deterministic tie order. Require disjoint one-step best-action sets.
  This deliberately constructed age distribution may be outside training support;
  keep its conclusion separate from naturally collected states.
- **Position:** change exactly one live drone's position by one cardinal cell.
  Keep every other state field fixed. Preserve grid bounds, charger ownership,
  occupancy and a three-move return margin for the moved drone. Consider all valid
  variants; require at least two common feasible actions and disjoint one-step
  best sets on that intersection. Choose the variant with the largest mean absolute
  centred change in gain. This is a local counterfactual, not a rollout transition.
- **No-op:** clone the same state without changes; all scores/ranks must be identical.

Compare each pair over the **intersection** of the two feasible-action masks.
Also record actual per-state masked argmax; a changed deployment action with an
unchanged common-set ranking can be caused by the mask. Record maximum score delta,
maximum centred score delta (subtract each state's common-set mean), first-argmax
change, and the fraction of action-pair orders changed (ties created/broken count).
Strict order flips exclude pairs tied in either state and report their denominator.
First argmax breaks score ties by ascending action ID; reference gain best sets
use absolute tolerance 1e-12. Do not compare absolute Q scales across recipes.

Aggregate pair fractions equally per eligible snapshot, not by pooling every
action pair. Report eligibility counts and both sides' informative-state regret
and best-set match rates. Controls and raw vectors permit independent recomputation.
A nonzero Q delta alone does not demonstrate useful coordination. Order changes
alone can occur in an untrained network; useful response should also improve
immediate-gain alignment. This diagnoses these particular checkpoints and selected
states; it cannot prove that representation, budget or exploration is the sole
cause, establish generalisation, or justify 40×40 scaling.

Publish this protocol and runner before executing it. Archive the prelaunch source
hashes, checkpoint identities, every case/mask/intervention and raw score vector,
per-state/pair metrics, complete denominators and checksum inventory. No model or
reward/representation change follows automatically from the measurements.
