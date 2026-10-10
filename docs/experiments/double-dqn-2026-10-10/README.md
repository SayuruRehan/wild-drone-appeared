# Corrected masked Double DQN: 20×20 development screen

The six study-integrity fixes were merged in PR #22. This follow-up checks whether
the corrected Double DQN option improves coordination. It does not justify 40×40 scaling.
See [the protocol](../../double-dqn-pilot-protocol.md), including the launch deviation.

Two masked DQN fits: 100 episodes, training seed 0, worlds 0–99, learning rate 0.001,
identical complete development environment and other training settings. Only the
Double DQN flag differs. Each final checkpoint was evaluated on worlds 5600–5619,
alongside masked random and Greedy (20 missions per policy).

| Policy | Mean return | Mean freshness | Drone losses |
|---|---:|---:|---:|
| Standard masked DQN | -11.1353 | 0.1378 | 0 |
| Corrected masked Double DQN | -10.5512 | 0.1574 | 0 |
| Masked random | -11.0199 | 0.1554 | 0 |
| Greedy | -4.5168 | 0.3183 | 0 |

Double DQN gained 0.5841 return and 0.0196 freshness over standard DQN. Relative
to masked random, the gains were only 0.4687 return and 0.0020 freshness. The
predefined gate requires at least 0.5 return and 0.01 freshness over both references,
plus zero losses. **The gate failed.** The relative gain over standard DQN is one
development fit, not evidence of robust learned coordination or an RL advantage.
The candidate will not be scaled to 40×40 on this evidence.

## Evidence and limits

- `screening.csv`: all 80 final-policy/baseline missions; `paired.csv`: 40 scenario-paired
  comparisons. `summary.csv` and `gate.json` contain the aggregates and gate calculation.
  Return SD in this pilot is across scenarios, not across training seeds.
- `raw/`: both complete 100-row training logs, 20-row final-validation logs, metadata
  and summaries. `standard.json` and `double.json` are the input configurations;
  the CLI overrides episodes and validation settings as shown below.
- `identities.json`: source and checkpoint hashes. The restored source snapshot was
  verified against the published code; runner metadata has no local git revision.
  Binary checkpoints are excluded from git. Exact checkpoint replay requires those
  binaries; the archived results can be checked independently without them.
- The launch used the screening worlds as validation with interval 100. There was
  one validation checkpoint, at the final episode, and no choice among checkpoints.
  This deviation is recorded explicitly; these worlds are development evidence.
  Neither the earlier final-test split nor the reserved confirmation split was used.
- After both fits finished and wrote their exports, the runner's parent evaluation
  was interrupted. Screening was executed separately with one PyTorch CPU thread;
  only those separately completed screening outputs are reported here.

## Reproduce

From the repository root at source revision `4abd41c694541855fcb4d6b27b87dd0d81123d9c`
(or the results commit, whose learning source is identical), install the RL/dev extras.
For each input configuration, run the existing masked-study runner:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_masked_study.py \
  --config docs/experiments/double-dqn-2026-10-10/standard.json \
  --episodes 100 --training-seeds 0 --algorithms dqn --jobs 1 \
  --validation-interval 100 \
  --validation-seeds 5600 5601 5602 5603 5604 5605 5606 5607 5608 5609 5610 5611 5612 5613 5614 5615 5616 5617 5618 5619 \
  --out runs/double-dqn-reproduction/standard
```

Repeat with `double.json` and a separate `double` output folder. The runner's
`final masked dqn`, `masked random` and `greedy` comparison rows correspond to this
screen. Ignore its selected-policy rows for this gate. Compute scenario means and
subtract each reference from Double DQN for return and weighted freshness; apply
the thresholds in the protocol; `gate.json` records their pass/fail outcomes.
Timing is machine-dependent.

The next improvement should address inspection/coordination learning directly.
Changing the TD target rule alone has not produced a useful gap over safe random
actions. Retain the fresh confirmation split and require repeated-seed gains before
the proposed 40×40 retraining study.
