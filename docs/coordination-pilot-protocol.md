# Training-only coordination potential: bounded 20×20 probe

Hypothesis: a dense state-based signal for approaching stale priority cells can
help a masked learner coordinate before inspection. This is one candidate, not
a claim that reward sparsity is the proven cause of the remaining performance gap.

Use the unchanged complete development environment: 20×20, horizon 160, battery
80, recharge 5, sensor radius 1, inspection deadline 25. Compare corrected masked
Double DQN with shaping weight 0 and 10. Freeze all other settings: learning rate
0.001, gamma 0.99, replay 2000, batch 32, warmup 128, target update 100, epsilon
1→0.05 over 10000 steps, spatial encoding on. Both fits use 100 episodes,
training seed 0 and training worlds 0–99. Monitoring worlds 5000–5004 are separate
from screening worlds **5700–5719**; monitoring interval 100. Use final checkpoints
only. Evaluate masked random and Greedy on the same screening worlds. Do not use
the old final-test worlds 10000–10019 or the previous screen 5600–5619 for tuning.

For each cell, staleness is min(age/deadline, 1), freshness is 1-staleness, and
distance is minimum Manhattan travel to any live drone's square sensor footprint.
The potential is the priority-weighted mean of
`freshness + 0.5*staleness*exp(-distance/3)`; a cell receives no duplicate drone
credit. Priority weights include current burning/frontier state exactly as in
the environment. The two constants are fixed before this screen.

Train on `original_reward + weight*(gamma*Phi(next)-Phi(current))`, setting Phi=0
on every terminal transition (including early fleet loss). This follows
[Ng, Harada and Russell (1999)](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).
Its discounted sum differs from the original discounted return only by the
initial-state constant `-weight*Phi(initial)`. It preserves that discounted
objective; it guarantees neither better approximation nor improved undiscounted
mission return. Training targets use shaping, but evaluation, checkpoint selection
and logged mission returns use the original environment reward. No potential is
computed by deployed policies. No environment or action-mask rule changes.

Pilot gate: shaped Double DQN must improve mean original mission return by ≥0.5
and mean priority-weighted freshness by ≥0.01 over BOTH unshaped Double DQN and
masked random, with zero losses in every candidate mission. Greedy is a context
baseline, not a gate threshold. These thresholds match the previous practical
screen; they are not significance tests. A failed screen stops this candidate's
scale-up. Archive failures, do not retune weight/constants on this screen.

If the pilot passes, compare both recipes with seeds 0, 1, 2 and 300 episodes each
on fresh confirmation worlds 5800–5819 (monitoring 5000–5004). Require the same
aggregate gains, positive scenario-paired mean return gains over both references
for each training seed, and zero candidate losses. Only that confirmation permits
a separately designed 40×40 retraining study. A one-seed screen cannot establish
generalisation or an RL advantage.

Archive complete input configurations, training and monitoring CSVs, summaries,
all screening rows and scenario-paired differences, gate calculation, source and
checkpoint hashes, and elapsed time. Binary checkpoints stay outside git. The
runner writes the frozen launch manifest before fitting and refuses an existing
output directory. CPU threading is one per fit and for evaluation.
