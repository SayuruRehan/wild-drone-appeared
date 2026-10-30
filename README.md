# Wildfire Drone Fleet — COMP6008

A reinforcement-learning project for **persistent wildfire inspection using three battery-constrained drones**. One central controller chooses a joint move for all three drones on a simulated 40×40 fire grid, and the drones share a single charging station. The task is modelled as a fully observable, finite-horizon Markov decision process (MDP).

**Live demo:** https://sayururehan.github.io/wild-drone-appeared/ (mission replays and results)

**Team:** Lithma Perera, Sulakna Weerasinghe and Sayuru Bopitiya (Curtin University, COMP6008 Reinforcement Learning, 2026).

## Where to start

| You want to… | Open |
| --- | --- |
| See the full project (problem, MDP, agents, experiments, results) | [`Wildfire_Drone_Fleet_clean.ipynb`](Wildfire_Drone_Fleet_clean.ipynb) |
| Read the written report | [`report/main.pdf`](report/main.pdf) |
| Watch missions and browse the results | the [live demo](https://sayururehan.github.io/wild-drone-appeared/) |
| See exactly what was submitted as code on 23 Oct | [`COMP6008_Wildfire_Drone_Fleet/`](COMP6008_Wildfire_Drone_Fleet/) |

## Results in brief

DQN, REINFORCE and A2C were each trained with three seeds (300 episodes, 90,000 steps per run) and evaluated on **20 unseen worlds** against a one-step Greedy controller, a rolling-horizon planner, a lawnmower sweep and random policies. With a battery feasibility mask, no learned policy lost a drone in any of the 180 evaluation missions. Inspection quality, however, stayed at the level of random feasible actions: weighted freshness was **0.084** for DQN and A2C and 0.069 for REINFORCE, against 0.084 for masked random and **0.221** for Greedy. The notebook and report explain why and propose improvements.

The simulator uses illustrative fire, wind and energy settings, not calibrated real-world wildfire physics.

## Experiments

The notebook reads its settings and results from these folders. The main-study and follow-up folders each have a README describing the protocol, the data and how to rerun it.

| Experiment | Folder |
| --- | --- |
| Settings for the 20×20 development study and the 40×40 main study | [`configs/`](configs/) |
| 20×20 development: masked vs unmasked agents and the tuning screen | [`docs/experiments/final-2026-10-05/`](docs/experiments/final-2026-10-05/) |
| 40×40 main study: training logs and evaluation on 20 unseen worlds | [`docs/experiments/rl-main-2026-10-11/`](docs/experiments/rl-main-2026-10-11/) |
| Rolling-horizon planner, learning-rate sweep and sensitivity study | [`docs/experiments/rl-followups-2026-10-11/`](docs/experiments/rl-followups-2026-10-11/) |

The other folders in `docs/experiments/` hold the earlier diagnostic experiments (DQN stability, imitation of Greedy, shaping and scaling pilots) summarised in Section 8 of the notebook.

## Repository structure

```text
wild-drone-appeared/
├── Wildfire_Drone_Fleet_clean.ipynb   Main project notebook, with saved outputs
├── README.md                          This file
├── requirements.txt, pyproject.toml   Python dependencies and package definition
│
├── src/wild_drone/                    Python package used by the notebook
│   ├── environment.py                 Gymnasium wildfire environment (MDP, fire spread, batteries)
│   ├── config.py                      Environment and training settings
│   ├── baselines.py                   Greedy, sweep, random and masked-random controllers
│   ├── learning.py                    DQN, REINFORCE and A2C
│   ├── masked_learning.py             Feasibility mask for the learners
│   ├── experiments.py, cli.py         Study runners and command-line entry point
│   ├── webapp/                        Local simulation server for the Mission Control UI
│   └── *_probe.py, diagnostics.py …   Diagnostic experiments from development
│
├── configs/                           Experiment settings (20×20 development, 40×40 main study)
├── docs/
│   ├── experiments/                   Archived results of every experiment (one folder per study)
│   │   ├── final-2026-10-05/          20×20 masked vs unmasked study and tuning screen
│   │   ├── rl-main-2026-10-11/        40×40 main study: training logs and evaluation
│   │   └── rl-followups-2026-10-11/   Planner, learning-rate sweep and sensitivity study
│   ├── *-protocol.md, *-plan.md       Design notes written before each experiment
│   └── mission-control.md             How the web UI works and how to run it locally
│
├── report/                            IEEE report (LaTeX sources and compiled main.pdf)
│   ├── sections/                      Parts I–III and the appendix
│   └── figures/                       Figure 1 and the script that draws it
│
├── web/                               Mission Control web app (React + Vite), deployed as the live demo
├── scripts/                           Command-line scripts that ran and audited the experiments
├── tests/                             pytest suite for the package
├── experiments/sula/                  Exploratory analysis scripts (reward signal, action landscape)
├── notebooks/                         Earlier Colab notebooks used for GPU runs
├── .github/workflows/                 CI: tests, report build, Pages deploy, experiment runners
│
├── COMP6008_Wildfire_Drone_Fleet/     Self-contained copy of the code submitted on 23 Oct
├── COMP6008_Wildfire_Drone_Fleet.zip  The same folder, zipped as uploaded
└── *_DeclarationOfOriginality.pdf     Declarations of originality of the three members
```

## Running the notebook

Use **Python 3.11 or 3.12**. From the repository root:

```bash
python -m pip install -r requirements.txt
python -m jupyter notebook Wildfire_Drone_Fleet_clean.ipynb
```

In Jupyter, choose **Restart Kernel and Run All Cells**. By default the notebook loads the archived results and runs a few short demonstrations, so it finishes in minutes. Full training and the longer reruns are switched off; you can enable them with the flags in Section 1. Keep the folder structure intact, because the notebook reads `configs/` and `docs/experiments/` through relative paths.

The trained main-study checkpoints are not stored in the repository. See [`docs/experiments/rl-followups-2026-10-11/README.md`](docs/experiments/rl-followups-2026-10-11/README.md) to download them for checkpoint-based reruns.

## Running the tests

```bash
python -m pip install -e .
python -m pytest -q
```

## Building the report

```bash
cd report
latexmk -pdf main.tex
```

You need a TeX distribution with IEEEtran. Pushing a change under `report/` to `master` makes CI rebuild `main.pdf` automatically; see [`report/README.md`](report/README.md).

## Running the web UI locally

The live demo needs no setup. To run it locally with the interactive Simulation Lab, follow [`docs/mission-control.md`](docs/mission-control.md). In short, you need Python 3.11+ and Node 22.12+, then run `wild-drone-ui` for the simulator service and `npm run dev` in `web/`.
