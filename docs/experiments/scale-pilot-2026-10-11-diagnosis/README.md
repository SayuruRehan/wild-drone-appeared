# Saved 40×40 mission diagnosis

This is a **post-hoc development diagnostic**, following the frozen pilot's
negative mission result. It does not replace the original capability gates,
select a fitted seed, add training, or open confirmation/final-test worlds.
All results use the original artifact from Actions run `38111928600`, source
`7c9562aa6064f46d96cc9a163de3deb2ecaa3e19`.

## What completed

Three networks were fit from scratch for 2,000 updates each. All three passed
the frozen ranking gate and failed the full-mission gate. The run contains
140 complete missions and 42,000 steps. This is supervised inspection scoring;
there were **zero RL updates**.

| Policy | Mean inspection freshness | Mean overdue fraction | Mean return | Drone losses |
|---|---:|---:|---:|---:|
| Centered scorer, across three seed means | 0.192003 | 0.697632 | -17.844649 | 0 |
| Greedy, across 20 worlds | 0.226995 | 0.643557 | -15.649919 | 0 |
| Masked random, across 20 worlds | 0.086493 | 0.885049 | -25.408448 | 0 |

Each scorer seed's freshness was only 83.4–85.4% of Greedy, below the frozen
90% threshold. Each also exceeded Greedy's overdue fraction plus 0.05. Every
scorer seed beat masked random and passed the loss/fleet-failure conditions.
No collisions were recorded for these policies. Energy use was similar for
the scorer and Greedy (779.12 versus 786.70 per mission), so low activity or
fleet loss alone does not explain the performance gap.

The gap appears throughout the mission:

| Steps | Scorer freshness | Greedy freshness |
|---|---:|---:|
| 1–100 | 0.200134 | 0.225036 |
| 101–200 | 0.165188 | 0.203952 |
| 201–300 | 0.210687 | 0.251996 |

These are descriptive means from saved traces, not independent observations or
statistical significance tests. The learned group contains 60 missions;
Greedy contains 20 matching world IDs.

## Check rankings on the scorer's own trajectories

Replay the saved actions on all 20 development mission worlds for each seed.
At the original snapshot times 30/60/120/240, inspect the **pre-action state**.
Check its observation hash, saved feasible mask, chosen action and replayed
reward. Use all 125 cloned simulator actions to compute the original one-step
inspection-gain reference. This produces 80 states per seed, 240 overall.
Read the existing score vectors; do not recompute or fit predictions.

| Seed | Original fresh-collector regret | Own-trajectory regret | Own-trajectory best-action match |
|---|---:|---:|---:|
| 0 | 0.099251 | 0.076217 | 31.25% |
| 1 | 0.090526 | 0.077356 | 33.75% |
| 2 | 0.099589 | 0.085199 | 35.00% |

All 240 sampled states are informative. The sampled own-trajectory rankings
show **no deterioration in average normalized regret relative to the fresh
collector packet**. This does not rule out errors between snapshots or prove
that training coverage is sufficient. It does argue against calling the
observed mission gap a demonstrated collapse on the policy's own states.

The scorer still chooses a one-step suboptimal action in roughly two thirds
of these snapshots. Around elapsed step 120, best-action match is only
10%, 5%, and 20% for seeds 0, 1, and 2. Greedy scores immediate inspection
coverage directly, with a small movement tie-break; the model approximates
that reference. Small repeated ranking errors, coverage trajectories and
return/charging decisions are therefore plausible explanations to investigate,
rather than established causal findings. The policies' emergency-mask
difference remains disclosed in the frozen protocol.

## Continuation

Retain this as a negative 40×40 pilot. Before another fit, design a separate
development experiment that distinguishes approximate ranking errors from
return/charging and longer-horizon coordination effects, especially around
the middle of the mission. Keep the current result, thresholds, seeds and
checkpoints fixed. Do not claim that DQN, REINFORCE or A2C improved from this
supervised result.

## Reproduction

Download and extract `frozen-40x40-pilot-7c9562aa6064f46d96cc9a163de3deb2ecaa3e19`
from the original Actions run, then run from the repository root:

```bash
python docs/experiments/scale-pilot-2026-10-11-diagnosis/diagnose-saved-missions.py \
  --run runs/scale-pilot-2026-10-11 --out runs/new-scale-diagnosis
```

`on-policy-ranking.csv` holds all 240 state rows. `on-policy-summary.json`
records the grouping and hashes of the exact trace/score inputs.
`mission-trace-summary.json` records the mission and phase means shown above.
The full frozen-run integrity/replay audit is in the adjacent
`scale-pilot-2026-10-11` evidence archive.
