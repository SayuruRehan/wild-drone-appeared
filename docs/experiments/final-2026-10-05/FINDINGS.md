# Findings (5 Oct 2026 study)

## 1. Has the agent solved the problem?

Partly. **Masking solved safety but not coordination.**

On the 20 test scenarios, every masked agent kept the whole fleet alive in every mission
(0 losses). Their returns are -10.9 (DQN), -11.2 (A2C) and -11.5 (REINFORCE). The unmasked
agents trained the same way scored -24.6 to -65.4 and lost 1.7 to 3 drones per mission,
which is no better than random play (-29.0).

The masked agents are still far behind the planning baselines: greedy (-5.3), the
rolling-horizon planner (-5.2) and even sweep (-11.9) are close to or better than them.
They are about the same as **masked random (-11.4)**. So most of the improvement comes
from the mask, and the learned part adds little.

## 2. Why the learned part adds little

- **Tiny per-step signal.** Across feasible actions, the best and worst immediate reward
  differ by only 0.019 on average, and the best and second-best by 0.0017. Under masked
  random play, the mission return varies by 1.06 (SD) between scenarios. A Monte Carlo
  learner (REINFORCE) has to find a 0.02 signal under a noise level about 50 times larger.
  (`experiments/sula/reward_signal.py`)
- **Short training budget.** 300 episodes is about 48,000 steps. DQN in the literature
  uses millions of steps [Mnih et al. 2015].
- **Single 125-way head.** Each joint action is learned separately; similar actions share
  no structure.
- **Tuning did not change the picture.** All 18 tuning runs ended between -10.1 and -12.2
  on validation, within the masked-random spread (SD 0.9).

## 3. Sensitivity (test scenarios, one training seed)

| Condition | Masked DQN | Masked random | Greedy | Planner |
|---|---:|---:|---:|---:|
| default | -10.9 | -11.4 | -5.3 | -5.2 |
| battery 60 | -10.5 | -11.1 | -4.6 | -5.2 |
| battery 120 | -12.1 | -10.8 | -4.3 | -5.1 |
| recharge 3 | -11.6 | -11.6 | -6.5 | -6.6 |
| recharge 8 | -10.4 | -10.9 | -4.8 | -4.4 |
| spread 0.02 | -10.7 | -11.6 | -5.7 | -5.4 |
| spread 0.08 | -11.1 | -11.6 | -6.5 | -5.2 |

- **Slow charging hurts every controller most**: greedy goes from -5.3 to -6.5. With one
  charger, recharge time is the coordination bottleneck.
- **Fast fire spread hurts greedy (-6.5, 0.3 losses) but not the planner (-5.2, 0 losses)**:
  looking ahead pays off when hazards change quickly.
- Greedy loses a drone in some test missions (0.15 per mission), the planner never does.

## 4. Implications

The fully observable problem is solvable by planning in milliseconds, so an RL agent must
beat a strong, cheap baseline to be useful here. The main lesson is that safe exploration
(the mask) is necessary before any learning can happen, and that the reward needs a stronger
per-step coordination signal than it has now.
