# Reference experiments: planner, sensitivity and learning-rate sweep (40x40)

Three follow-up experiments to the frozen 40x40 main study, covering what the proposal
promised and the frozen study left out: the rolling-horizon planner comparison, sensitivity
to battery / charging / fire spread, and a check on the untuned learning rates.

This folder is reference code. It is listed in `.git/info/exclude`, so git ignores it. It
reads the repository (`src/`, `configs/rl-main-frozen.json`,
`docs/experiments/rl-main-2026-10-11`) and writes only inside `docs/experiments/rl-followups-2026-10-11/`. The frozen
archive is not modified.

## Files

| File | What it does |
|---|---|
| `common.py` | Shared helpers: loads the frozen config, flies one mission and returns the frozen `episodes.csv` columns, loads a trained checkpoint as a masked-argmax policy, runs missions across CPU cores, builds mean / SD tables. |
| `01_planner_eval.py` | Runs `RollingHorizonPolicy` on the 20 frozen final worlds (22000-22019) and tabulates it next to the frozen baselines and learners. First re-runs the four frozen baselines to prove the harness reproduces the archive. |
| `02_sensitivity.py` | Changes one setting at a time (battery capacity, recharge rate, spread probability) and re-flies greedy, planner, masked random, sweep and the nine frozen learner checkpoints. No retraining. |
| `03_lr_sweep.py` | Trains each algorithm from scratch at three learning rates (one seed, 300 episodes) and evaluates the final weights on validation worlds 23000-23019. |
| `results/` | All outputs (CSV, PNG, the trained sweep checkpoints). The checkpoints (`policy.pt`) are not committed; `results/lr_sweep/SHA256SUMS` lists their hashes. |
| `checkpoints/` | The nine frozen learner checkpoints downloaded from the `scale-rl` workflow run. |

## How to run

Python 3.12 with the repository's pinned versions. From `docs/experiments/rl-followups-2026-10-11/`:

```bash
python3.12 -m venv .venv
.venv/bin/pip install numpy==2.4.6 gymnasium==1.4.0 matplotlib==3.11.2 pillow==12.3.0 torch==2.5.1
```

The learner checkpoints are GitHub Actions artifacts of the frozen run (kept 90 days from
2026-10-11). `02_sensitivity.py` checks each file's SHA-256 against the frozen
`fit-*.json` and skips any learner whose checkpoint is missing.

```bash
for a in dqn reinforce a2c; do for s in 0 1 2; do
  gh run download 38134714083 -R SayuruRehan/wild-drone-appeared \
    -n frozen-40x40-rl-855e5510ff46b16082ab7d551e7c228acab6f572-$a-$s -D checkpoints/$a-$s
done; done
```

Then:

```bash
.venv/bin/python 01_planner_eval.py --check      # 2-world timing, ~45 s
.venv/bin/python 01_planner_eval.py --workers 5  # ~3.5 min
.venv/bin/python 02_sensitivity.py --workers 8   # ~16 min
.venv/bin/python 03_lr_sweep.py --workers 9      # ~20 min
```

`03_lr_sweep.py` does not retrain a fit whose `policy.pt` already exists, but it first checks
that the file is the finished fit it would train (same algorithm, learning rate, seed,
environment and 300 completed episodes) and stops if it is not. It then reports how many
checkpoints are byte-identical to the published ones in `results/lr_sweep/SHA256SUMS`.
Retraining gives different weights on a different machine (see the last point under the
learning-rate results), so a retrained checkpoint will not match those hashes.

To reproduce the committed sweep results exactly without retraining, download the nine
published checkpoints from the `lr-sweep-checkpoints-2026-10-11` release and check them:

```bash
gh release download lr-sweep-checkpoints-2026-10-11 -R SayuruRehan/wild-drone-appeared
tar -xzf lr-sweep-checkpoints-40x40.tar.gz -C results/lr_sweep
(cd results/lr_sweep && shasum -a 256 -c SHA256SUMS)
```

`02_sensitivity.py --plot-only` and `03_lr_sweep.py --plot-only` rebuild the summary and
figure from the saved episode CSVs. Runtimes are from a 10-core Apple-silicon Mac.

## Conventions (same as the frozen study)

- Metrics use full-horizon accounting: if the whole fleet is lost early, the missing steps
  count as zero freshness and fully overdue.
- Learners act by argmax over the feasible actions given by the shared battery / collision
  mask. Random and masked random are seeded with the world number, and so is the planner.
- Learner mean = average over worlds within each training seed, then over the 3 seeds; SD
  is across the 3 seed means. Baseline and planner SD is across the 20 worlds.
- No significance tests. Twenty worlds and three seeds do not support them.

## Results

### 1. Rolling-horizon planner (`results/planner_vs_frozen_table.md`)

Harness check: greedy, sweep, random and masked random reproduce the archived per-world
numbers to within 2e-15 (`results/harness_reproduction.csv`), so the planner rows are
directly comparable with the frozen rows.

| Policy | Freshness ↑ | Overdue ↓ | Return ↑ | Losses ↓ |
|---|---:|---:|---:|---:|
| dqn | 0.0844 ± 0.0025 | 0.8687 | -24.79 ± 0.25 | 0 |
| reinforce | 0.0690 ± 0.0073 | 0.8940 | -25.81 ± 0.59 | 0 |
| a2c | 0.0843 ± 0.0116 | 0.8695 | -24.83 ± 0.92 | 0 |
| greedy | 0.2208 ± 0.0266 | 0.6556 | -16.10 ± 1.50 | 0 |
| **planner (new)** | 0.2207 ± 0.0269 | 0.6668 | -16.55 ± 1.49 | 0 |
| sweep | 0.0649 ± 0.0154 | 0.9012 | -26.07 ± 0.81 | 0 |
| random | 0.0596 ± 0.0178 | 0.9210 | -42.87 ± 1.12 | 3 |
| masked_random | 0.0838 ± 0.0243 | 0.8860 | -25.46 ± 1.33 | 0 |

- The planner matches greedy on freshness and is slightly worse on return and overdue.
  Paired on the same worlds, planner minus greedy return is -0.45 (SD 1.18); the planner is
  better in 7 of 20 worlds. Lookahead of 4 steps with 8 candidates buys nothing over the
  one-step greedy rule at the default setting.
- Both are far ahead of every learner: about 2.6 times the freshness of DQN and A2C.
- Cost: 67 ms per decision for the planner against 1.1 ms for greedy, measured serially on
  the same machine (`results/planner_check.json`), about 60 times more. One mission takes
  about 20 s. The `decision_ms` column in `planner_episodes.csv` (154 ms) was measured with
  5 parallel workers and is inflated; frozen-archive timings come from a GitHub runner.
- Planner settings are the class defaults (horizon 4, 8 candidates, 2 sampled futures).
  Other settings were not tried.

### 2. Sensitivity (`results/sensitivity_summary.csv`, `results/sensitivity.png`)

Mean weighted freshness over the 20 final worlds (learners: mean of 3 seed means):

| Condition | greedy | planner | masked random | sweep | dqn | reinforce | a2c |
|---|---:|---:|---:|---:|---:|---:|---:|
| default | 0.2208 | 0.2207 | 0.0838 | 0.0649 | 0.0844 | 0.0690 | 0.0843 |
| battery 120 | 0.2049 | 0.2083 | 0.0834 | 0.0665 | 0.0964 | 0.0829 | 0.0939 |
| battery 200 | 0.2151 | 0.2160 | 0.0877 | 0.0643 | 0.0777 | 0.0608 | 0.0753 |
| recharge 5 | 0.2025 | 0.2068 | 0.0780 | 0.0638 | 0.0805 | 0.0656 | 0.0803 |
| recharge 12 | 0.2258 | 0.2274 | 0.0872 | 0.0654 | 0.0848 | 0.0710 | 0.0849 |
| spread 0.02 | 0.2080 | 0.2162 | 0.0804 | 0.0657 | 0.0844 | 0.0687 | 0.0841 |
| spread 0.08 | 0.2082 | 0.2139 | 0.0821 | 0.0656 | 0.0856 | 0.0662 | 0.0842 |

Mean return:

| Condition | greedy | planner | masked random | sweep | dqn | reinforce | a2c |
|---|---:|---:|---:|---:|---:|---:|---:|
| default | -16.10 | -16.55 | -25.46 | -26.07 | -24.79 | -25.81 | -24.83 |
| battery 120 | -17.76 | -17.13 | -25.53 | -26.01 | -24.01 | -24.98 | -24.22 |
| battery 200 | -16.46 | -16.87 | -25.19 | -26.07 | -25.22 | -26.41 | -25.44 |
| recharge 5 | -17.29 | -17.37 | -25.79 | -26.14 | -25.03 | -26.09 | -25.09 |
| recharge 12 | -15.70 | -16.04 | -25.27 | -26.04 | -24.73 | -25.67 | -24.79 |
| spread 0.02 | -17.60 | -16.73 | -25.63 | -26.02 | -24.77 | -25.83 | -24.84 |
| spread 0.08 | -18.24 | -16.76 | -25.55 | -26.02 | -24.67 | -26.00 | -24.83 |

- **Greedy and the planner remain the strongest group across all tested conditions.** They
  stay at 0.20-0.23 freshness in every condition; all learners, masked random and sweep
  stay at 0.06-0.10. Rankings inside each group do change: greedy leads the planner on
  freshness only at the default, and DQN, A2C and masked random swap places from one
  condition to the next.
- **Greedy loses the whole fleet in 4 of the 140 missions, and only away from the default**:
  battery 120 (world 22003), spread 0.02 (world 22008), spread 0.08 (worlds 22000 and
  22019). In each case the three drones arrive low on battery two cells from base, every
  joint move toward base collides, `candidate_actions` falls back to "all stay", and they
  hover for 16 steps until the batteries empty. This is the deadlock described in the
  `action_mask` docstring; the masked policies avoid it through `emergency_return_mask`,
  and the planner avoided it in all 140 of its missions. No other policy lost a drone.
- Those four failures explain most of greedy's drop in return. On the missions greedy
  completes intact its return is -16.70 (battery 120), -16.59 (spread 0.02) and -16.03
  (spread 0.08), close to the default -16.10. So the planner's advantage under changed
  spread (-16.7 against -17.6 and -18.2) is robustness to this failure, not better
  inspection.
- Faster charging helps greedy and the planner a little (0.221 to 0.226-0.227); slower
  charging costs them about 0.015-0.018 freshness.
- A smaller battery makes the fixed learners look better (DQN 0.084 to 0.096) and a larger
  one worse (0.078). The weights are unchanged and battery input is normalised by capacity,
  so this is most likely the mask forcing earlier returns and more passes near the base,
  not learned behaviour. That explanation was not tested.
- Spread probability has almost no effect on the learners, masked random or sweep.
- Learner rows measure robustness of fixed weights. They say nothing about what each
  algorithm could learn if trained under the changed setting.

### 3. Learning-rate sweep (`results/lr_sweep_summary.csv`, `results/lr_sweep.png`)

One training seed (0), 300 episodes, final weights, 20 validation worlds 23000-23019:

| Algorithm | Learning rate | Freshness ↑ | Overdue ↓ | Return ↑ |
|---|---:|---:|---:|---:|
| dqn | 3e-5 | 0.0869 | 0.8681 | -24.71 |
| dqn | 1e-4 (frozen) | 0.0981 | 0.8490 | -24.00 |
| dqn | 3e-4 | 0.0804 | 0.8791 | -25.19 |
| reinforce | 3e-4 | 0.0715 | 0.8898 | -25.66 |
| reinforce | 1e-3 (frozen) | 0.0799 | 0.8812 | -25.22 |
| reinforce | 3e-3 | 0.0840 | 0.8682 | -24.80 |
| a2c | 3e-5 | 0.0839 | 0.8669 | -24.70 |
| a2c | 1e-4 (frozen) | 0.0835 | 0.8664 | -24.72 |
| a2c | 3e-4 | 0.0825 | 0.8690 | -24.84 |
| greedy | - | 0.2258 | 0.6445 | -15.65 |
| masked_random | - | 0.0906 | 0.8779 | -25.14 |

- **None of the tested rates closed the gap to greedy in these seed-0 runs.** The best fit
  reaches 0.098 freshness against greedy's 0.226.
- Some fits do edge past masked random. DQN at its frozen rate is ahead on both freshness
  (0.0981 against 0.0906) and return (-24.00 against -25.14), and six of the nine fits
  have a better return than masked random. The freshness gap is smaller than the SD
  across worlds (about 0.015).
- One training seed cannot establish that any of these differences is reliable, and it
  cannot rule out the learning rate as a contributing cause of the shortfall. What the
  sweep shows is that a 3-fold change either way was not enough on its own.
- REINFORCE improves steadily with a larger rate. A2C is flat across a 10-fold range.
- With one seed per setting, differences of 0.01 between rates are within the seed spread
  seen in the frozen study (SD across seed means 0.003-0.012). Treat the ordering of rates
  as suggestive only.
- DQN's training return improves over the first 100 episodes at every rate (about -25.5 to
  -24) and then plateaus; REINFORCE and A2C show no trend.
- The frozen-rate fits do not retrace the archived seed-0 training curves on this machine
  (`reproduces_frozen_training` is False), although the initial weights are bit-identical
  to the archived `initial.pt`. The archived fits ran on a GitHub Linux runner and these on
  Apple silicon; a floating-point difference between the two is the likely cause but was
  not confirmed. The sweep is therefore a same-machine comparison of three rates, not an
  extension of the archived fits.

## Runtimes (10-core Apple-silicon Mac)

| Step | Work | Wall time |
|---|---|---:|
| Planner check | 2 worlds, serial | 40 s |
| Planner evaluation | 80 baseline + 20 planner missions, 5 workers | about 3.5 min |
| Sensitivity | 1,820 missions, 8 workers | 15.8 min |
| Learning-rate sweep | 9 fits in parallel, then 220 validation missions | 18.9 min training (DQN 1,127 s, A2C 192 s, REINFORCE 185 s per fit) |

## Not done

- No learner was retrained under the changed battery / charging / spread settings.
- The sweep used one training seed and three rates per algorithm; no other hyperparameter
  was varied and no checkpoint selection was used.
- Planner horizon, candidate count and sample count were left at their defaults.
- No 60x60 runs.
