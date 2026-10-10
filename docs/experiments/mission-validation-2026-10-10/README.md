# Fixed inspection scorers in full 20×20 missions

Broader training coverage improved full-mission inspection quality at the same
fixed model and fitting budget. All three expanded-data models passed both
prospective mission data-effect and Greedy-relative capability checks. This
validates the improvement beyond one-step ranking on these development worlds;
it does not demonstrate trained RL, 40×40 performance or a final-test result.

## Frozen experiment

Source/protocol were published before rollout at
`558c31d1c01b99c20bca5b4141fac813325a8531` in PR #29, based on PR #28 merge
`227f125d5cff1447aa39bba660cd1253e27d5d4c`.
See [the prospective protocol](../../mission-validation-protocol.md).
All parent archive Git blobs/checksums, source hashes, and six final checkpoint
identities/metadata were verified before launch. The models retain their
original supervised format, centered inputs, seeds 0/1/2 and 2,000 completed
updates. No fitting or RL updates occurred in this experiment.

Configuration is exactly the preceding 20×20 environment: horizon 160,
battery 80, recharge 5, radius 1, inspection deadline 25 and max fuel 100,
with all other fields in `launch.json`. Every policy ran on fresh development
worlds 6600–6619, disjoint from training 0–199 and previous development worlds
6200–6215/6400–6415. Confirmation 5800–5819 and final-test 10000+ remain unopened.
No world was dropped. The 11 conditions completed all 220 episodes and 35,200
steps, taking 130.8 seconds on one CPU thread; no GPU was required.

Small/expanded models and their two fixed training-action-mean controls use
first-index argmax under the unchanged feasible action mask and its emergency
return fallback. Native Greedy/Sweep use their original candidate-action rule,
which lacks that fallback; masked-random uses the common mask and a fresh RNG
seeded by each world. These rules are disclosed rather than attributing safety
to the learned scores alone. The primary small/expanded comparison shares the
same feasibility rule. Scores remain inspection predictions, not Q values;
deployment uses observations and the mask without labels or simulator lookahead.

## Mission results

Network entries are mean ± sample SD across three fitted-seed means, each based
on the same 20 worlds. Greedy/control/baseline SDs are across 20 worlds, a
different unit. All freshness and overdue metrics use the full 160-step horizon;
the runner preserves zero freshness/full overdue padding for failed fleets.

| Metric | Small scorer | Expanded scorer | Native Greedy |
| --- | ---: | ---: | ---: |
| Weighted freshness ↑ | 0.217784 ± 0.011220 | 0.293318 ± 0.005806 | 0.316081 ± 0.028001 |
| Overdue fraction ↓ | 0.652901 ± 0.015563 | 0.538289 ± 0.010736 | 0.504810 ± 0.035106 |
| Original mission return ↑ | −8.133517 ± 0.395989 | −5.478288 ± 0.223666 | −4.689387 ± 0.785230 |
| Energy used | 381.229 ± 3.400 | 386.488 ± 2.672 | 393.250 ± 14.060 |
| Safe returns | 3.533 ± 0.076 | 3.250 ± 0.132 | 3.750 ± 0.851 |
| Charge completions | 3.483 ± 0.076 | 3.183 ± 0.144 | 3.650 ± 0.745 |
| Drone losses | 0 | 0 | 0 |
| Fleet failures | 0 | 0 | 0 |
| Collision attempts | 0 | 0 | 0 |

Expanded freshness improved 34.7% over small, and overdue inspections fell
11.46 percentage points. Expanded freshness was 92.8% of Greedy's mean. Energy
use increased about 1.4% while freshness increased; fewer charge completions
alone are not a claim of better charging. Greedy still had higher freshness and
lower overdue fraction. The expanded overdue fraction remains 53.8%, so passing
the declared relative benchmark does not establish sufficient inspection of
every cell or a deadline guarantee. No statistical significance claim is made.

| Seed | Small freshness | Expanded freshness | Freshness difference | Small overdue | Expanded overdue | Both decisions |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.230696 | 0.299930 | +0.069234 | 0.635156 | 0.525979 | Pass |
| 1 | 0.212253 | 0.289054 | +0.076800 | 0.659311 | 0.545710 | Pass |
| 2 | 0.210403 | 0.290972 | +0.080568 | 0.664235 | 0.543177 | Pass |

Every seed improved mean freshness by ≥.01 with no increase in overdue, losses
or fleet failures. Every seed reached ≥90% of Greedy freshness, with overdue
≤Greedy+.05, mean losses ≤Greedy+.1 and fleet failures ≤Greedy. `completion.json`
records each per-seed decision separately; `episodes.csv` and `per-fit.csv`
record all 11 policy/seed conditions. There were no losses, fleet failures or
collisions in any of the 220 episodes, and all fleets remained intact.

| Contextual policy | Freshness | Overdue fraction |
| --- | ---: | ---: |
| Small training-action-mean | 0.111768 | 0.814272 |
| Expanded training-action-mean | 0.113162 | 0.816029 |
| Sweep | 0.113429 | 0.822464 |
| Masked-random | 0.137019 | 0.803819 |

## Evidence and reproduction

The archive preserves every episode, per-fit/aggregate row, matched seed/world
difference, and every step's action, pre/post observation hashes, full reward
components, freshness, overdue and resource events. Eight lossless NPZ files
contain all 125 float32 scores and boolean masks for every network/control
decision: 19,200 network vectors and 6,400 control vectors. Baseline decisions
are reproduced from their controllers and world-seeded RNGs. Initial models
were not evaluated and no checkpoint was selected using these results.

`recompute.py` independently sums all 35,200 step records into the 220 episode
metrics, checks all 11 fits, 84 aggregates, 60 paired rows and each separate
decision, including exact row-key coverage. `replay.py` runs the original
simulator for every recorded action, independently reproduces each model/control
score and argmax, and checks masks, observation hashes, reward components,
resource events and unchanged checkpoint identities before/after evaluation.
Both audits report PASS. Replay verified 27 frozen source files and found no
native baseline action outside the common mask in these worlds.
Timing is excluded from exact replay equality. Verification outputs are
archived separately. `trajectory-diagnostics.csv` describes unique positions,
grounded/stationary steps and visited inspection cells from replay; these are
exploratory descriptions, not tuned policy choices or extra promotion criteria.
`checksums.json` covers all other archive files.

From the frozen source checkout and original raw parent checkpoints:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_mission_validation.py \
  --parent-root runs/training-coverage-2026-10-10 \
  --out runs/mission-validation-2026-10-10 \
  --source-revision 558c31d1c01b99c20bca5b4141fac813325a8531
python docs/experiments/mission-validation-2026-10-10/recompute.py
PYTHONPATH=src python docs/experiments/mission-validation-2026-10-10/replay.py \
  --parent-root runs/training-coverage-2026-10-10
```

The runner requires a fresh output directory. The first command regenerates a
run under `runs/`; the two audit commands verify
the checked-in archive with the original parent checkpoints, rather than that
new run's files. Raw supervised model binaries remain under ignored `runs/`,
with their original identities archived. Repository
validation passed: 208 tests, five CUDA-only skips, Ruff and the CLI demo.
The result supports designing a separate 40×40 pilot with appropriate training
and mission baselines. It does not automatically transfer this configuration
or these checkpoints to 40×40; the learned scorers remain supervised policies,
and the assignment's RL algorithms still require separate validation.
