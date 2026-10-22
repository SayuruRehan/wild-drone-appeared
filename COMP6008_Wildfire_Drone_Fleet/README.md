# Wildfire Drone Fleet — COMP6008

A reinforcement-learning project for **persistent wildfire inspection using three battery-constrained drones**. One controller selects joint actions on a simulated fire grid while the drones share a single charging station. The environment is formulated as a fully observable, finite-horizon Markov decision process (MDP).

## Main notebook

**[Wildfire_Drone_Fleet_clean.ipynb](Wildfire_Drone_Fleet_clean.ipynb)** is the project notebook. It contains:

- The problem formulation, environment dynamics, state/action spaces and reward design.
- Implementations of **DQN, REINFORCE with a value baseline, and synchronous A2C**, including pseudocode-to-code mappings.
- Conventional baselines and the battery/collision feasibility mask.
- Experiment code, hyperparameter tuning, learning curves, held-out evaluation, planner and sensitivity studies, and a critical review.

The notebook includes saved outputs. By default, it loads archived experiments and runs small verification examples; full training and lengthy experiment reruns are **disabled** unless their flags in Section 1 are enabled.

## Setup and execution

Use **Python 3.11 or 3.12**. From the repository root:

```bash
python -m pip install -r requirements.txt
python -m jupyter notebook Wildfire_Drone_Fleet_clean.ipynb
```

In Jupyter, select the Python environment containing these dependencies, then choose **Restart Kernel and Run All Cells**. Keep the repository folder structure intact: the notebook reads its configuration and archived results through relative paths.

To verify the standalone implementation and tests:

```bash
python -m pip install -e .
python -m pytest -q
```

## Experiments and results

The final 40×40 study trained **three seeds per RL algorithm** (nine fits, 300 episodes each) and evaluated their final policies on **20 unseen simulation worlds**. The feasibility mask prevented drone loss in all 180 learned evaluation missions, but learned inspection freshness remained near masked random (approximately **0.084** for DQN and A2C), below Greedy (**0.221**). The negative result and its limitations are examined in the notebook.

The data and settings used by the notebook are in:

- [`configs/`](configs/) — development and frozen experiment settings.
- [`docs/experiments/final-2026-10-05/`](docs/experiments/final-2026-10-05/) — historical 20×20 comparison and tuning.
- [`docs/experiments/rl-main-2026-10-11/`](docs/experiments/rl-main-2026-10-11/) — frozen 40×40 training and evaluation records.
- [`docs/experiments/rl-followups-2026-10-11/`](docs/experiments/rl-followups-2026-10-11/) — planner, learning-rate sweep and sensitivity scripts, results and instructions for obtaining optional trained checkpoints.

The environment and learning package is in [`src/wild_drone/`](src/wild_drone/), with supporting [`scripts/`](scripts/) and [`tests/`](tests/). **The main-study trained checkpoints are not included in the repository checkout**; see [the follow-up experiment instructions](docs/experiments/rl-followups-2026-10-11/README.md) for retrieving them when reproducing the full checkpoint-based sensitivity runs.

The simulation uses illustrative fire, wind and energy settings rather than calibrated real-world wildfire physics.
