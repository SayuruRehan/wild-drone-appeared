# 20×20 position and urgency ranking diagnosis

**Both learned networks respond to the inputs, but their preferred actions rarely
respond to changed inspection opportunities.** The shaped network changed its
common-feasible-set preferred action in none of the 28 urgency or 27 position
pairs. The unshaped network changed it in two pairs of each kind. This supports
weak decision sensitivity in these checkpoints; it does not establish that the
representation cannot encode coordination or that training budget is the sole cause.

The [protocol](../../ranking-diagnostic-protocol.md), runner and implementation
were published on `sr` at `2da69d657d0fca901c28af8e35a25e4285311e73` before
execution. No policies were trained, selected or changed. These are the exact
final 100-episode, training-seed-0 corrected masked Double DQN checkpoints from
the [coordination pilot](../coordination-2026-10-10/). The complete 20×20
development environment is unchanged. No confirmation or 40×40 worlds were opened.

## Input response

All 32 planned snapshots were collected: worlds 5900–5907, masked-random and
Greedy collectors, elapsed steps 16 and 80. The models did not collect or choose
the states. Urgency pairs change the location of stale inspection ages while
preserving their histogram and priority-weighted global freshness. Position pairs
translate one live drone by one cardinal cell, preserving every other state field.
Pairs are chosen from input/gain information without consulting any model scores.
Both interventions require disjoint one-step freshness-gain best sets.

| Policy | Urgency preferred-action changes | Urgency mean pair-order changes | Position preferred-action changes | Position mean pair-order changes |
|---|---:|---:|---:|---:|
| Unshaped Double DQN | 2/28 | 0.237% | 2/27 | 0.289% |
| Shaped Double DQN | 0/28 | 1.331% | 0/27 | 0.378% |
| Untrained network | 0/28 | 0.157% | 0/27 | 0.386% |
| Fixed state-blind scores | 0/28 | 0% | 0/27 | 0% |
| One-step inspection reference | 28/28 | 73.218% | 27/27 | 28.106% |

Preferred actions and order changes in this table use the intersection of the
two feasible masks. Action-pair order changes include ties created or broken;
fractions are averaged equally over eligible snapshots. Strict-flip counts and
their denominators are also in `pairs.csv`. The reference's complete preferred
action switching is expected from the pair eligibility requirement; it confirms
the intended input contrast, not long-term policy superiority.

Both learned models had nonzero centred score changes in every eligible pair and
some order changes.
The shaped network's extra urgency order changes did not change its top action.
State dependence alone is insufficient evidence of useful coordination: an
untrained network also has order changes. Absolute Q magnitudes are not compared
between recipes.

Actual deployment actions, using each state's own safety mask, changed in 3/27
unshaped position pairs, 0/27 shaped, 2/27 untrained and 1/27 state-blind. The
fixed-score control shows why deployment changes alone cannot demonstrate a
learned position response. All no-op vectors, rankings and preferred actions were
exactly unchanged across 28 eligible no-op pairs for all five policies.

Four Greedy step-80 states had only one feasible action and were ineligible for
pair comparisons. A fifth state, world 5900/masked-random/step 80, had no valid
one-cell position contrast with a shared disjoint optimum. Every exclusion is
logged in `coverage.csv`; no planned snapshot was missing.

## Is the response useful for immediate inspection?

Normalized regret is `(best_gain-selected_gain)/(best_gain-worst_gain)` on the
comparison's feasible set: zero selects a one-step best action, one a worst
action. Uniform-random expectations use that same feasible set. Zero gain-range
states are excluded from these means. Natural states are distinct from the
constructed counterfactuals.

| States | Informative states | Unshaped regret | Shaped regret | Uniform-random expected regret |
|---|---:|---:|---:|---:|
| Natural | 28/32 | 0.603140 | 0.538528 | 0.604007 |
| Urgency pair sides | 56/56 | 0.573973 | 0.573448 | 0.553155 |
| Position pair sides | 54/54 | 0.587383 | 0.492518 | 0.584876 |

Each learned model selected a one-step best action in only 1/28 informative
natural states. The shaped policy's lower natural and position regret is a
descriptive gain-alignment signal. It did not translate into better mission
return/freshness than unshaped in the preceding pilot. Neither model beat the
uniform-random expected regret on the constructed urgency pairs. The raw and
summary files include reference ties, selected actions and all control results.

These results concern two particular checkpoints sharing one training seed and
selected development states. Constructed urgency maps can be outside training
support; position probes are local counterfactuals, not natural rollout steps.
Immediate inspection gain omits long-term coverage, energy and charging value.
Consequently these measurements do not prove an architecture defect or identify
budget, reward or exploration as the sole failure. There is no new improvement
claim and the previous failed scale-up gate remains in force.

Earlier [Greedy imitation](../imitation-2026-10-03/),
[fitting-budget](../fitting-budget-2026-10-04/) and
[derived navigation/energy input](../representation-2026-10-04/) probes already
showed poor held-out decisions and autonomous flights. Repeating those label or
budget experiments is not the recommendation here.

A useful next diagnostic is to fit the existing representation to exact one-step
inspection targets on a development-only state split, then test held-out action
rankings. Targets would cover all feasible actions and tied best sets, rather than
one Greedy command per state. This isolates inspection opportunity learning from
teacher tie choices and autonomous battery-return errors. Success would show
capacity for this narrower target under direct supervision; failure would still
not prove representational impossibility. This archive does not execute that
experiment or change the representation.

## Evidence and reproduction

- `launch.json`: prelaunch source hashes, full environment, checkpoint identities,
  seeds and published source revision.
- `completion.json`, `coverage.csv`: planned/observed counts, eligibility and runtime.
- `natural.csv`, `pairs.csv`, `summary.csv`: unrounded measurements and denominators.
- `raw/<case>/state.json`: masks, natural observation hash and exact interventions.
- `raw/<case>/<policy>.json`: all 125 scores for every available state variant.
- `verification.json`: implementation and evidence checks.
- `independent-audit.json`: independent metric recomputation and full state replay.
- `checksums.json`: SHA-256 and byte sizes for every other archive member.

The 192 raw JSON files permit independent recomputation of the metrics without
binary weights. Deterministic input-state reconstruction uses the frozen source,
collector/world/step and stored observation hash. Learned score replay additionally
requires the original checkpoint binaries, whose identities are in `launch.json`;
binaries are not committed. Keep the original coordination-pilot checkpoint folder
when rerunning:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/diagnose_rankings.py \
  --pilot-root runs/coordination-pilot-2026-10-10 \
  --source-revision 2da69d657d0fca901c28af8e35a25e4285311e73 \
  --out runs/ranking-diagnostic-replay
```

Run from a checkout of that source revision; the output directory must not exist.
The diagnostic completed in 23.24 seconds on one CPU thread. A GPU was not needed.
