# Exact inspection targets: strong fitting, weak held-out rankings

**The existing network fits the training inspection targets, but its ranking
quality transfers poorly to new worlds. All three fits failed the predeclared
capability criterion.** Training regret is almost zero and mean tied-best match
is 92.74%; held-out regret is 0.4624 and match is 7.80%. This is a supervised
capability diagnostic, not an RL or autonomous-mission improvement.

The [protocol](../../inspection-target-protocol.md) and implementation were
published at `2ff35318b67ceaa8a7bd186b85650095771e083d` before collection/fitting.
Three fresh copies of the unchanged FleetNetwork (seeds 0/1/2) each received
2,000 Adam updates on one shared corpus. Physics, observations, nine spatial
channels, 24-vector values, 125 outputs and action feasibility were unchanged.
No prior weights, reward change, representation change, checkpoint selection,
RL update or budget extension was used. Held-out predictions were read only
after all three fixed fits finished. No larger-grid or autonomous run followed.

## What was fitted

Natural states were collected by independent masked-random and Greedy policies
on training worlds 0–39 and held-out development worlds 6000–6015, at elapsed
steps 16/32/64/96. Every feasible action received an exact original one-step
inspection-gain target from an isolated environment clone. Targets were
normalized within each state to [0,1], preserving tied best actions; loss was
squared error averaged within each state's feasible actions, then across states.
This differs from earlier studies of one Greedy label per state.

There are 319/320 observed training states (303 informative) and 127/128
held-out states (124 informative). Greedy terminated before step 96 on training
world 20 and held-out world 6000. Those missing snapshots are logged, with no
invented post-termination states. Sixteen training and three held-out states
had zero feasible gain range and were excluded from fitting/informative means.
World splits are disjoint; the previous diagnosis, confirmation and final-test
worlds were not opened in this experiment.

## Training and held-out quality

Normalized regret is `(best_gain-selected_gain)/(best_gain-worst_gain)`: zero
selects a one-step best action, one a worst action. Match uses the full tied-best
set, rather than agreement with one teacher command. Means below average
informative observed states within each fit; SD is sample SD across three fits,
not across independent states or worlds.

| Policy | Training regret | Held-out regret | Training best match | Held-out best match |
|---|---:|---:|---:|---:|
| Initial networks, mean ± fit SD | 0.5994 ± 0.0564 | 0.5781 ± 0.0627 | 3.74% ± 1.33 pp | 4.57% ± 2.83 pp |
| Fitted networks, mean ± fit SD | 0.0014 ± 0.0005 | 0.4624 ± 0.0082 | 92.74% ± 2.31 pp | 7.80% ± 1.86 pp |
| State-blind training-action mean | 0.4678 | 0.5032 | 9.24% | 6.45% |
| Uniform random, analytical expectation | 0.6038 | 0.6036 | 4.42% | 5.25% |
| Exact one-step inspection reference | 0 | 0 | 100% | 100% |

The state-blind control averages each action's normalized target over informative
training states where it is feasible; it uses no held-out targets. It is one
shared control, not three independent fits. The exact reference uses target
information and confirms gain alignment, not long-term optimality.

| Fit seed | Final training regret | Final held-out regret | Held-out best match |
|---|---:|---:|---:|
| 0 | 0.001880 | 0.464116 | 8.87% |
| 1 | 0.001470 | 0.469648 | 5.65% |
| 2 | 0.000825 | 0.453477 | 8.87% |

The trained models improve held-out regret over initialization by 0.1157 on
average, and over the action-mean control by only 0.0408. All three satisfy the
training-regret criterion <= .05; none satisfies held-out regret <= .20,
improvement >= .15 over the shared action-mean control, or best match >= .25.
Only seed 2 improves over its own initialization by >= .15. The complete
predeclared checks are in `completion.json`; no threshold was changed afterward.

Masked regression error also separates fit from transfer: final training
0.000297 ± 0.000237, held-out 0.127106 ± 0.008783. The state-blind control's
held-out error is lower, 0.088349. These errors compare scores to normalized
supervised targets; they are not Q-value calibration or mission returns.

## Position and urgency responses

All 32 planned secondary cases were collected on held-out worlds 6000–6007,
both collectors, steps 16 and 64. The previous model-blind rules constructed
32 balanced urgency and 32 one-cell position contrasts before fitting. All
had at least two shared feasible actions and disjoint reference best sets.
No contrast entered training or the primary capability criterion.

| Fit seed | Initial → final urgency preferred-action changes | Initial → final position preferred-action changes |
|---|---:|---:|
| 0 | 1/32 → 5/32 | 0/32 → 9/32 |
| 1 | 0/32 → 4/32 | 3/32 → 14/32 |
| 2 | 0/32 → 8/32 | 0/32 → 10/32 |

These changes use the common feasible set. Increased preferred-action
responsiveness does not establish that the response is useful: final urgency
regrets are 0.5647/0.5773/0.5723. Natural held-out, position and artificial
urgency results remain separate. Artificial age distributions may be outside
training support. The state-blind common-set rankings never change, inspection
reference tops change in all contrasts by construction, and all initial/final
and control no-op vectors/rankings are exactly unchanged.

## What this changes about the diagnosis

This network has capacity to fit the narrow target on these training examples
under this budget. The large training/held-out gap is consistent with poor
generalization under the current corpus and inductive bias. It does not identify
representation, data coverage, optimization, collection distribution or RL signal
as the sole cause, and does not establish a universal architectural limitation.
Simply extending fitting time is not the first supported remedy: training
ranking fit is already very strong. More CPU/GPU compute alone was not tested.

Read this alongside the previous [imitation](../imitation-2026-10-03/),
[fitting-budget](../fitting-budget-2026-10-04/) and
[navigation/energy input](../representation-2026-10-04/) studies; their fits are
context, not extra replicates. The present supervision removes arbitrary teacher
tie choices but still shows a substantial held-out decision gap.

A next bounded comparison could test drone-local inspection information or
action-relative inspection features against this unchanged input transform,
using matched seeds, source observations, target corpus and fitting budget.
That would test a spatial inductive-bias hypothesis rather than presume a fix.
Only a useful held-out result would support considering an RL experiment;
mission improvements and repeated-seed validation would still be needed before
40×40 scaling. This archive implements no such representation change.

## Evidence and reproduction

`launch.json` binds the prelaunch source, full environment and fixed recipe.
`datasets.json` and `checkpoints.json` bind all three tensor datasets and six
initial/final supervised checkpoints by byte size and SHA-256. Original binary
observation datasets and weights remain in ignored local run folders; these
explicitly supervised exports are rejected by the RL loaders.

`inputs.json.gz` contains every input case, observation hash, all 125 action
gains and masks. Each `seed-N/{initial,final}-scores.json.gz` contains all
unrounded 125-action vectors, including natural and counterfactual states.
`probe-plan.json` records exact intervention locations, pairing and eligibility.
`natural.csv`, `pairs.csv`, `per-fit.csv` and the three 2,000-row training logs
preserve every measured numeric record. Gzip JSON can be read with Python's
`gzip` and `json` modules; compression does not round values.

`recompute.py` independently regenerates scalar selection/tie metrics and
action-pair orders, without importing production metrics or fitting functions.
`metric-verification.json` records that check; `replay-verification.json`
records full collector/counterfactual/target and score replay.
`verification.json` records tests/provenance, and `checksums.json` binds all
other archive members. Metric recomputation requires the archived vectors;
score replay additionally requires the original ignored binaries.

From a checkout of the frozen source revision, use a fresh output directory:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_inspection_targets.py \
  --source-revision 2ff35318b67ceaa8a7bd186b85650095771e083d \
  --out runs/inspection-targets-replay
```

From the final repository checkout, audit numeric results with:

```bash
python docs/experiments/inspection-targets-2026-10-10/recompute.py
```

The complete collection, three fits and evaluation took 163.94 seconds on one
CPU thread. No Colab/GPU run was required.
