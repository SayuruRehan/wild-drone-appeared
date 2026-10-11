# Controlled exact-inspection reference at 40×40

The frozen supervised pilot passed ranking checks but failed full missions. The
saved-state diagnosis found no average ranking collapse on sampled own-policy
states. This separate, pre-run diagnostic asks whether removing approximation
error can close the mission gap under the scorer's existing action rules.

Run one deterministic exact-inspection reference on all original development
worlds 7000–7019, 20 complete 300-step missions. Reuse the original environment,
source, rewards, common feasible mask including emergency return, and first-index
argmax. Score every action with the original `gain_scores` / `transition_coverage`
reference: 125 independent one-step simulator clones, actual post-transition
live-drone footprints, pre-fire inspection urgency. Clones must not change the
real state or RNG. Use full precision reference gains, without model fitting,
normalization rounding or Greedy's movement preference. This is simulator-assisted
diagnostic lookahead, not a deployable model, trained RL, or a long-term oracle.

Compare against the frozen archived missions of all three centered scorer seeds,
native Greedy and masked random on the same worlds. Do not select a fitted seed,
retrain, change masks, or replace the prior pilot result. Native Greedy retains its
original candidate rule and small movement penalty; both differences remain
disclosed. Preserve original freshness/overdue/return/resource metrics, complete
per-world paired differences, every action/observation hash/reward component and
all full-precision score/mask vectors. Record runtime separately; clone lookahead
is much more expensive than network inference. No new worlds, fits or RL updates;
confirmation 5800–5819 and final-test 10000+ are outside this diagnostic.

Interpretation is descriptive, not a significance test or new promotion gate:
if exact rankings reach the original 90%-Greedy freshness and overdue benchmarks,
approximation error is sufficient to explain an attainable performance gap under
this action rule; return/charging constraints do not alone prevent reaching that
benchmark. If they do not, more fitting alone is not supported; investigate the
objective, ties and trajectories. Neither outcome proves a unique causal mechanism
or a useful trained RL policy. Preserve the negative original pilot unchanged.

Freeze and publish this protocol, runner and tests before rollout. Check frozen
parent source and archive hashes before launch. Independently replay all missions,
reconstruct masks and full score vectors, check the first-index argmax, verify
actual cloned inspection rewards, and recompute full-horizon episode/paired means
without production metric helpers. Test state/RNG immutability, actual rewards in
resource/collision states, and first-index ties. Publish audited results, pass the
repository tests/lint/demo and CI, review and merge sr→master, synchronize sr.
