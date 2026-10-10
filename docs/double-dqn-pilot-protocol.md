# Bounded 20×20 Double DQN probe — 10 October 2026

Question: does correcting masked Double DQN help coordination, compared with
standard masked DQN under the same action mask, physics, reward and training budget?
This is an algorithm ablation, not a promise that the fix improves performance.

Use the complete development environment (20×20, horizon 160), frozen DQN
learning rate 0.001, 100 training episodes, seed 0, training worlds 0–99,
validation worlds 5000–5004, and final checkpoints. Compare standard and Double
DQN; every other TrainConfig field is identical. Evaluate both, masked random
and Greedy on development worlds 5600–5619. The old final-test worlds 10000–10019
are not used for this probe or for choosing improvements.

Pilot gate: Double DQN must improve mean return by at least 0.5 and mean
priority-weighted freshness by at least 0.01 over BOTH standard DQN and masked
random, with zero losses in every evaluated mission. These are practical screening
thresholds, not statistical significance claims. A failed pilot stops this candidate's
scale-up. Report the outcome even if it is negative.

If the pilot passes, compare both recipes with three training seeds (0, 1, 2),
300 episodes each and new development worlds 5800–5819. Require the same mean
gains, a positive paired mean return gain for every training seed, and zero losses.
Only passing that confirmation permits a 40×40 study.

For 40×40, retrain on configs/main.json rather than silently transferring a
20×20 checkpoint. Retain the three RL methods and conventional baselines, freeze
settings using validation only, and use a new final-test split. Grid, horizon,
sensor radius and battery changes must all be stated; this is a new domain.

Archive raw training/validation/evaluation CSVs, summaries, exact configurations,
source and checkpoint hashes, and the gate calculation. Binary checkpoints remain
outside git. A one-seed pilot cannot establish generalisation or an RL advantage.

## Execution note

The initial protocol text was written before training (SHA-256
`75f184eacbf2564d371a9ab2c000ae50f34f675b77ccab1a786041ea28bc3b57`).
The CLI launch inadvertently used 5600–5619 as validation worlds with interval 100,
rather than 5000–5004. There is exactly one validation checkpoint, at episode 100;
only final policies are used for the gate. The worlds were not used to choose among
checkpoints or recipes. Nevertheless these results are explicitly a development
screen, not an independent held-out confirmation. The fresh 5800–5819 confirmation
split and the original screening thresholds remain unchanged.
