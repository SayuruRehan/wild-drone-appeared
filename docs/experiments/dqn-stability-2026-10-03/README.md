# Lower-rate DQN and Double DQN validation screen

All six fits completed the [predeclared 200-episode comparison](../../dqn-stability-plan.md).
Neither method learned useful inspection/recharge coordination. The short validation
rollouts did not reproduce the excessive Q values seen in the original T4 run, but this
does not establish stability at 1,000 episodes, value calibration, or an RL advantage.

Both arms use learning rate 0.00003, training seeds 0/1/2, training scenarios 0–199,
validation scenarios 5000–5004, and otherwise identical development settings. Standard
DQN remains the default. The target-rule split is the only learning difference between
arms. Validation every 20 episodes selects over ten boundaries; the selected results
are biased. Final test scenarios 10000+ were not used.

Each fitted model is averaged over five scenarios before calculating mean and sample
SD across three fits. Baseline SD is across five scenarios, a different unit.

| Policy | Return mean ± SD | Freshness mean | Fleet losses mean | Blocked attempts mean |
|---|---:|---:|---:|---:|
| Greedy | −4.922 ± 0.588 | 0.313 | 0.000 | 0.000 |
| Planner | −5.024 ± 0.656 | 0.312 | 0.000 | 0.000 |
| Sweep | −12.142 ± 0.499 | 0.110 | 0.000 | 0.000 |
| Random | −28.337 ± 0.797 | 0.097 | 3.000 | 2.600 |
| Final DQN | −38.016 ± 6.901 | 0.052 | 3.000 | 43.800 |
| Selected DQN | −29.333 ± 0.886 | 0.068 | 3.000 | 3.600 |
| Final Double DQN | −32.548 ± 2.961 | 0.060 | 2.933 | 19.600 |
| Selected Double DQN | −26.863 ± 2.048 | 0.063 | 2.400 | 4.333 |

The Double DQN sample means are less negative, but the small fit count, shared validation
set, short training budget and selection bias do not justify declaring a winner. Both
final means are worse than random; all fits are far behind greedy. The selected policies
inspect little and frequently lose drones. A low TD loss does not mean a useful controller.

## Value and checkpoint checks

All 12 portable exports matched authoritative episode counts, metadata and exact tensors.
All 60 diagnostic mission returns, freshness, losses and collision counts exactly replayed
the corresponding comparison rows. Checkpoint hashes bind diagnostics to the verified bytes.

| Export group | Maximum observed Q | Bound violations / visited states |
|---|---:|---:|
| Final DQN | −0.0083 | 0 / 1,785 |
| Selected DQN | 1.0351 | 0 / 1,789 |
| Final Double DQN | 17.6289 | 0 / 1,854 |
| Selected Double DQN | 1.1835 | 0 / 2,056 |

The upper bound is `freshness_reward * sum(gamma**t for t in range(remaining_steps))`.
Inspection gain is at most one and other reward terms are nonpositive. Diagnostics test
the largest of all 125 Q values at each visited state against its remaining-horizon bound.
No violations in these 7,484 visited states cannot rule out failures elsewhere or later.
The positive final Q of 17.63 in Double DQN seed 1 deserves monitoring in longer fits.

Selected episodes for seeds 0/1/2 were 140/120/180 for DQN and 160/40/160 for Double DQN.
The last 100 training episodes had median Huber losses around 0.14–0.15 in all fits.
This is substantially different from the original high-rate T4 loss explosion, but that
older study also used a longer budget and different runtime. It is historical context,
not a matched rate-effect control. The earlier one-fit lower-rate screen also is not
included as an extra independent fit or required to reproduce an identical trajectory.

## Reproduction and provenance

The README's two `run_validation_pilot.py` commands and `diagnose_dqn.py` command reproduce
this protocol. Completed runs remain locally under `runs/dqn-stability-v3`; they can resume
with a larger total budget in a separately declared follow-up experiment. Binary training
states and policies are excluded from Git; verified hashes and raw records are retained.

The run launched from a dirty working tree based on `f312eecc`, using Python 3.12.14,
PyTorch 2.5.1 CPU and NumPy 2.3.5. Per-fit metadata records the observed revision/runtime.
`source-manifest.json` records hashes of the training/runtime/environment files in use.
The launch version of the runner is preserved in `source-launch/`: the final runner adds
a preflight guard against changing a study's fit roster, without changing learning.
Both arms ran concurrently in separate directories with one Torch thread per process;
elapsed times are not a CPU/GPU or target-rule throughput benchmark.

Records comprise 1,200 training rows, 300 periodic validation rows, two 50-row comparisons
(the baseline rows are duplicated between arms), and 60 diagnostic replay rows.
`archive-manifest.json` checksums the copied records. `per-fit.csv` averages diagnostic
metrics by variant, training seed and selection; `summary.json` then aggregates those
three fitted-model means, using sample SD (`ddof=1`). Its Q maxima and violation counts
are maximum/sum across the same rows. Baselines come from the comparison CSVs.

Next research work should diagnose exploration, repeated blocked commands, learned battery
return behaviour and value targets before a larger 40×40 study. A separately fixed longer
budget is needed to assess whether the lower rate remains stable. Reward changes, shields
and architectural changes should remain explicit ablations, not score-driven fixes.
