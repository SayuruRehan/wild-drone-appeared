# 20×20 training-only coordination signal: negative pilot

**The predefined improvement gate failed.** The shaped candidate was worse than
unshaped Double DQN on both original mission return and freshness. It had a return
gain over masked random, but essentially the same freshness. No confirmation or
40×40 scale-up was launched. This is a one-seed development screen, not a final test.

The [protocol](../../coordination-pilot-protocol.md) and implementation were
published on `sr` at `503189cc66d137aefadebba933573a233d13bbb9` before fitting.
Two corrected masked Double DQN fits used 100 episodes, training seed 0, worlds
0–99, the complete development environment and identical settings except shaping
weight 0 versus 10. Monitoring worlds 5000–5004 were separate from the fresh
screening worlds 5700–5719. Only final checkpoints were evaluated for this gate.
There was no launch deviation, checkpoint choice or candidate retuning.

| Policy | Mean original return | Scenario return SD | Mean freshness | Losses |
|---|---:|---:|---:|---:|
| Shaped masked Double DQN | -10.7591 | 0.6115 | 0.1485 | 0 |
| Unshaped masked Double DQN | -10.4835 | 0.5727 | 0.1560 | 0 |
| Masked random | -11.3204 | 0.8302 | 0.1465 | 0 |
| Greedy | -4.7986 | 0.7135 | 0.3102 | 0 |

There are 20 complete missions per policy. SD is across scenarios, not training
seeds. Relative to unshaped Double DQN, shaping changed mean return by -0.2757
and freshness by -0.0074. Relative to masked random, changes were +0.5613 and
+0.0021. The gate required ≥0.5 return and ≥0.01 freshness over both references,
plus zero candidate losses. It therefore failed against both references.

## What changed and what the result establishes

The potential combines priority-weighted freshness with proximity to stale cells,
counting each cell only once across drones. Training rewards add
`10*(gamma*Phi(next)-Phi(current))`, with terminal potential zero. Environment
reward, action masking, network, optimizer and deployment are unchanged. Validation
selection and evaluation use the original reward. Training logs retain original
`return`/`episode_return` and report separate `learning_return`, `shaping_return`,
`discounted_shaping_return` and `initial_potential`.

The discounted shaping identity held over all 100 candidate training missions
with maximum absolute residual 2.67e-15. The new zero-weight 100-episode model
matched every tensor of the earlier corrected Double DQN fit exactly. A separate
small pre-change reference also matched every tensor and every pre-existing CSV
field. These checks support implementation fidelity, not a performance benefit.

Each fit took approximately two minutes on CPU (two fits ran concurrently).
Safety remained strong on the screened missions, while Greedy still achieved
much better inspection quality. This candidate does not demonstrate that a denser
signal solves coordination. It also does not identify training budget, network
representation or exploration as the sole remaining cause. Increasing GPU size or
moving to 40×40 is not supported by this result.

## Post hoc behavior check

`behavior.csv` contains nonintervening replays of all 80 screening missions. Every
original mission metric matched the screen exactly, excluding machine-dependent
decision timings. This diagnosis was run after the failed screen and did not tune
the candidate. See `behavior_protocol.json` and `diagnose.py`.

The shaped and unshaped policies used means of 22.6 and 24.2 distinct joint actions
per mission, versus 82.75 for masked random and 59.3 for Greedy. The dominant-action
fractions were 0.1244 and 0.1422; shaping did not produce a simple all-stay collapse.
Action diversity is descriptive and does not prove good coordination. A useful
next bounded diagnosis is whether the learned action ranking responds to local
inspection urgency and each drone's position, before changing representation or
training budget. The previous imitation probes should be considered before
introducing another teacher-assisted recipe.

## Archived evidence and reproduction

- `screening.csv`: 80 missions; `paired.csv`: 40 scenario-paired differences;
  `summary.csv` and `gate.json`: aggregates and the frozen gate outcomes.
- `raw/`: complete 100-row training logs and 5-row monitoring logs for each fit,
  metadata and training summaries. There are no incomplete fits in this archive.
- `unshaped.json`, `shaped.json`, `launch.json`: exact expanded input settings,
  prelaunch source hashes, world splits, checkpoint choice and launch timestamp.
- `checkpoints.json`: SHA-256 and byte sizes of final/selected checkpoint files.
  Binary weights are excluded from git; exact checkpoint replay requires those
  files or a reproduced fit. Selected checkpoints were not used in the screen.
- `verification.json`: tests, lint scope, compatibility and numerical checks;
  `artifacts.sha256.json`: checksum inventory for all other archive files.

The restored working snapshot was verified against the published Git tree, and
`PYTHONPATH` explicitly selected its `src` directory. Raw metadata has a null local
Git revision because the snapshot was not a local checkout; `launch.json` records
the independently verified published source revision. No historical archive was
changed. Five CUDA-only tests were skipped; all 176 other tests and the CI lint
scope (`src tests scripts`) passed. Broad notebook lint still reports 58 existing
style violations; notebooks were outside this change.

From the source revision above, install the RL/dev extras and run:

```bash
PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/run_coordination_probe.py \
  --source-revision 503189cc66d137aefadebba933573a233d13bbb9 \
  --out runs/coordination-reproduction
```

The output directory must be new. The runner writes the launch manifest before
fitting, then produces original-metric screening results and gate outcomes.
Optional post hoc diagnosis:

```bash
python docs/experiments/coordination-2026-10-10/diagnose.py \
  --run-root runs/coordination-reproduction
```

The fresh 5800–5819 confirmation split remains reserved. It was not opened after
this failed pilot. Forty-by-forty training remains conditional on a future
candidate's improvement and repeated-seed confirmation.
