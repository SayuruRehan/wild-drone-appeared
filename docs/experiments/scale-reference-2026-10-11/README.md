# Controlled exact-inspection reference at 40×40

The independent replay **passed**: 20 complete 300-step missions, 6,000 real
decisions and 750,000 cloned action rewards. Every saved full-precision score
vector, feasible mask and first-index chosen action was reconstructed exactly.
All observation hashes, rewards, episode summaries and 100 paired rows matched.
The maximum difference between reference gains and actual cloned reward sums was
1.735e-17. See [audit.json](audit.json) and the [frozen protocol](../../scale-reference-protocol.md).

## Result

| Policy | Inspection freshness ↑ | Overdue fraction ↓ | Mean return ↑ | Drone losses |
|---|---:|---:|---:|---:|
| Exact inspection reference | 0.226350 | 64.48% | -15.686822 | 0 |
| Learned centered scorer, mean of three seed means | 0.192003 | 69.76% | -17.844649 | 0 |
| Native Greedy | 0.226995 | 64.36% | -15.649919 | 0 |
| Masked random | 0.086493 | 88.50% | -25.408448 | 0 |

The exact reference reaches **99.72% of Greedy's freshness**, compared with
84.58% for the learned scorer. It improves freshness by 17.89% relative to the
scorer mean and closes 98.16% of its mean freshness gap to Greedy. Its overdue
fraction is just 0.12 percentage points above Greedy. It reaches the original
90%-Greedy freshness and Greedy-plus-0.05 overdue benchmarks descriptively;
this is not a new capability gate or promotion of a fitted policy.

Paired freshness is higher than the scorer in 58 of 60 seed/world comparisons,
and higher than random in all 20 worlds. It exceeds native Greedy in 7 of 20
worlds and is slightly lower on average. These are descriptive development
results, not independent seed/world observations or significance tests.

All reference missions finish at the full horizon with an intact fleet, no
fleet failures and no collision attempts. Mean energy use is 789.20, safe returns
3.55 and charge completions 3.45 per mission. The simulator-assisted reference
costs about 173.68 ms per decision on this machine; complete rollout wall time is
1,107.01 seconds. Timing includes machine/runtime effects and is not a matched
hardware benchmark against the archived neural policy.

## Controlled comparison and limits

The rollout uses only the original development worlds 7000–7019, the unchanged
40×40 environment and rewards, and the learned scorer's common feasible mask
including emergency return and its first-index tie rule. Each decision evaluates
125 isolated simulator clones using pre-fire inspection urgency and actual
post-transition live-drone footprints. The real state and RNG are untouched by
lookahead. Full float64 gains are used without model normalization rounding.

The three learned seeds, Greedy and random are the frozen parent pilot's archived
missions on these same worlds. No fitted seed is selected. Native Greedy retains
its original candidate rule and small movement penalty, so it is not identical
to the exact reference. The primary exact-versus-scorer comparison holds the
mask and tie rule fixed. Parent source/input hashes are recorded in [launch.json](launch.json).
Published rollout source: `be6f7ad61289b605a28843bc598390debbe587fc`.
Published independent-audit source: `c5588b46c09e749ee1ba7ca0ff49b63f7604b17b`.

Removing score approximation closes almost all of the mission gap under the
existing scorer action rule. Thus those return/charging rules do not by
themselves prevent near-Greedy performance, and improving decision rankings is
a supported next target. This does not prove a unique error mechanism: changed
actions also change coverage trajectories and subsequent return/charging
decisions. There is no newly trained model, deployable oracle or RL advantage.
The [original failed pilot](../scale-pilot-2026-10-11/README.md) remains unchanged.
There are **zero new fits, zero RL updates and zero new worlds**; confirmation
5800–5819 and final-test 10000+ were not used.

## Next bounded experiment

Freeze a separate 40×40 comparison of the existing masked gain-regression loss
against a decision-focused ranking loss. Hold the representation, archived
training data, three seeds, sample sequence, update budget, optimizer, mask and
tie rule fixed so that the training objective is the only intervention. Define
ties by the exact gains; avoid treating equivalent optimal actions as errors.
Measure normalized decision regret and complete matched development missions,
not regression loss alone. Publish the protocol before fitting, preserve both
negative pilots and retain the original mission benchmarks.

This is a recommendation for the next experiment, not an experiment already
run. If the change does not improve complete missions, stop that direction and
inspect representation/coverage errors before spending a larger training budget.
Actual DQN, REINFORCE and A2C comparison and reserved confirmation/final evaluation
still remain for the assignment; supervised diagnostics do not fulfill that scope.

## Reproduce and inspect

All result files, including compressed full score vectors and step traces, are
committed beside this report. [checksums.json](checksums.json) records every
archive file's byte count and SHA-256, excluding itself.

From the repository root, install the RL extra (the audit imports PyTorch) and
create the output directory:

```bash
python -m pip install -e '.[rl]'
mkdir -p runs
python scripts/audit_scale_reference.py \
  --run docs/experiments/scale-reference-2026-10-11 \
  --out runs/reference-replay-audit.json
```

Re-auditing evaluates all 750,000 cloned
transitions and can take several minutes. [comparison.csv](comparison.csv)
contains every seed mean; [episodes.csv](episodes.csv) contains all 20 reference
missions; [paired.csv](paired.csv) contains all 100 matched differences.
`scores.npz` holds all 6,000 × 125 float64 scores and boolean masks;
`steps.csv.gz` holds actions, observation hashes and every reward component.
[completion.json](completion.json) records work counts and rollout runtime.
