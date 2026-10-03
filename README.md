# Wild Drone Appeared

**Coordinated wildfire inspection with a battery-constrained drone fleet.**

One central controller manages three drones in a **fully observable MDP**. The complete fire,
fuel, terrain, wind, inspection-age, drone and charging state is available. Visits deliver
fresh close-range inspections of known hazards; the task is sustained inspection and
recharge coordination, rather than discovering an unknown fire map.

This is the starting implementation of the approved Group 7 project. Simulator mechanics,
baseline comparisons and all three RL training pipelines are implemented. Short runs are
integration checks, **not evidence that the agents have learned or outperform planners**.
Grid sizes and every fire/energy parameter are illustrative, uncalibrated simulation settings.

![40x40 simulator: true fire state, inspection freshness and drone batteries](docs/assets/simulator-preview.png)

## Run a visual mission

Python 3.11+ is required; Python 3.12 is used in CI. From the repository root:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
python -m wild_drone demo --config configs/main.json --policy greedy --out runs/demo
```

Open `runs/demo/demo.gif`. It displays fire/fuel/priority zones, inspection freshness, wind,
drone positions and battery levels. `evaluation.csv` and `metadata.json` accompany the demo.
Policies `sweep`, `random` and `planner` are also available. The planner is substantially
slower; use a shorter horizon for a quick check.

## Train and compare the three methods

Install PyTorch first (the following CPU command is sufficient for the starting experiments),
then install the RL extra. A compatible GPU build can be selected from the
[official PyTorch installer](https://pytorch.org/get-started/locally/); this starting trainer
executes on CPU.

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[rl,dev]'
python -m wild_drone train --config configs/development.json --algorithm dqn --seed 0 --out runs/dqn
python -m wild_drone evaluate --checkpoint runs/dqn/policy.pt --scenario-seeds 10000 10001 10002 --out runs/dqn-eval
```

Use `--algorithm reinforce` or `--algorithm a2c` to train the other methods. Each controls the
entire three-drone fleet. Checkpoints include the complete environment/training configuration.
Checkpoint evaluation requires the same config; cross-domain battery/fire interventions and
transfer across grid sizes are future experiment work, not silently supported evaluation.

| Method | Family | Implemented update |
|---|---|---|
| DQN | Deep value-based | Replay, target network, joint-action TD values |
| REINFORCE | Monte Carlo policy gradient | Complete episode returns, categorical policy |
| A2C | Actor–critic | n-step bootstrap, shared actor/value encoder |

The A2C implementation is a synchronous, single-environment n-step advantage actor–critic.
It does not use asynchronous workers. All three are model-free; actor–critic also belongs
to the broad policy-gradient family. Full-grid convolutional input and 125 joint actions are
shared across methods. There is no learned-policy action mask or safety shield.
REINFORCE weights reward-to-go by absolute-time discount and uses a fixed mission-horizon
normaliser; A2C also uses absolute-time weights with its rollout-length normaliser. The
configurable entropy term is a local policy regulariser, separate from mission reward.

An end-to-end **smoke check** (tiny task and deliberately insufficient training):

```bash
python -m wild_drone compare --grid 8 --horizon 12 --episodes 2 --training-seeds 0 1 --scenario-seeds 10000 10001 --warmup-steps 4 --batch-size 4 --planner-horizon 2 --planner-candidates 3 --planner-samples 1 --out runs/smoke
python -m wild_drone plot --csv runs/smoke/comparison.csv --out runs/smoke/comparison.png
```

For development experiments:

```bash
python -m wild_drone compare --config configs/development.json --training-seeds 0 1 2 --scenario-seeds 10000 10001 10002 --out runs/development
```

Start with 20x20. The main config is 40x40; a 60x60 evaluation should follow only after a
learning/runtime pilot. The main config's 200 episodes is a starting budget, not a justified
convergence threshold. Tune budgets and hyperparameters using separate validation seeds
(e.g. 5000–5999), freeze the choices, then use final test seeds (e.g. 10000+). The CLI rejects
evaluation seeds that overlap training scenarios; it cannot detect manual reuse of test results
for tuning. No long experiment is launched automatically by installing or running tests.

## Baselines and fair evaluation

| Baseline | Behaviour |
|---|---|
| Random | Uniform joint commands; a sanity check without a battery-return rule |
| Sweep | Partitioned serpentine patrol with recharge and return heuristics |
| Greedy | Jointly maximises immediate priority/inspection-age gain |
| Planner | Bounded rolling-horizon stochastic shooting with replanning |

Greedy/sweep/planner use explicit battery-aware proposal rules. These are part of their
controllers, not a universal safety guarantee or a shield supplied to RL. The planner knows
the simulator kernel but uses independent hypothetical futures, never the live RNG stream.
Its default budget is horizon 4, eight candidates, two samples; tune it fairly and report
decision time. It is an approximate planner, not an optimal baseline.

CSV reports mission return, priority-weighted freshness, overdue fraction, energy, losses,
safe returns, completed charges, blocked collision attempts and decision time. If every drone
is lost, the remaining mission counts as unserved and its overdue cost is charged. Quality
metrics use the whole mission horizon, including that remainder. An episode does **not**
finish merely because the current fire burns out: priority-area inspection continues.

Compare on identical held-out scenario seeds. Each scenario regenerates whole fuel, priority,
ignition and wind conditions. Repeat training with multiple seeds; retain per-seed rows.
Plots aggregate each fitted model over scenarios before showing variation across training
seeds. Baseline variation is across scenarios. Error bars are sample SD, not confidence
intervals or significance tests. Baseline and RL rewards can differ substantially in short
runs; pilot results must not be treated as scientific findings.

## Project structure and checks

```text
src/wild_drone/     environment, controllers, learning, CLI, rendering and experiments
configs/           versioned development and main settings
tests/             physical accounting, MDP contract, planner and learning checks
notebooks/         project walkthrough, demo, training and comparison commands
docs/              design, implementation plan and references
runs/              ignored local GIFs, CSVs, metadata and checkpoints
```

```bash
python -m pytest -q
python -m ruff check src tests
```

See [design](docs/design.md) for transition order, observation schema and reward semantics,
and [references](docs/references.md) for technical foundations. Before assignment conclusions
or paper claims, complete longer learning pilots, independent validation/tuning, repeated
held-out evaluation, reward/constraint ablations and simulator sensitivity checks. RL's
advantage and publication novelty remain questions to investigate.
