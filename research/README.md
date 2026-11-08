# research/ — publication study

Working title: **Fleet-size-generalising, interaction-aware reinforcement learning for
multi-UAV wildfire inspection with a shared charger.**

This folder is separate from the frozen COMP6008 assignment. It imports `src/wild_drone`
read-only and never changes it, `configs/` or `docs/experiments/`.

- [`PROTOCOL.md`](PROTOCOL.md): claims, hypotheses, controllers, splits, gates, statistics.
- [`results/`](results/): versioned results (per-episode CSVs and tables).
- [`colab/gpu_runner.ipynb`](colab/gpu_runner.ipynb): GPU runs on Google Colab.

## Layout

| Path | Contents |
|---|---|
| `fleetrl/env.py` | N-drone environment; identical to the frozen one at N = 3 |
| `fleetrl/feasibility.py` | battery/collision/charging feasibility factored into per-drone and per-pair rules |
| `fleetrl/selection.py` | exact maximisation of unary + pairwise scores over feasible joint actions (N ≤ 8) |
| `fleetrl/baselines.py` | Greedy, masked random, Assign (target-assignment heuristic) for any N |
| `fleetrl/features.py` | compact replay snapshots and batched drone-centric torch features, incl. overlap/charger pair features |
| `fleetrl/models.py` | joint dueling, VDN, DCG, structured, structured+ value networks |
| `fleetrl/train.py` | Double DQN, n-step, feasibility-constrained targets; `python -m fleetrl.train` |
| `fleetrl/evaluate.py` | one mission → study metrics |
| `scripts/headroom.py` | gate G1: non-learning controllers across fleet sizes |
| `scripts/pilot.py` | launch a (model × seed) training matrix in parallel; resumable |
| `scripts/evaluate_runs.py` | evaluate runs across fleet sizes next to the baselines |
| `configs/` | environments (copies of the frozen settings) and world-seed splits |
| `tests/` | equivalence with the frozen code, feasibility, exact selection, features, models |

## Quick start

```bash
python -m pip install -e . torch pytest
python -m pytest -q research/tests
```

```bash
export PYTHONPATH=research:src
python research/scripts/headroom.py --env dev20 main40 --drones 3 4 5 6 8 --worlds 50
```

```bash
export PYTHONPATH=research:src
python research/scripts/pilot.py --kinds joint vdn dcg structured --seeds 0 1 \
    --steps 100000 --env dev20 --parallel 4 --root research/runs/pilot-dev20 \
    -- --eval-every 25000 --eval-worlds 10 --eval-drones 3 5
```

```bash
export PYTHONPATH=research:src
python research/scripts/evaluate_runs.py --root research/runs/pilot-dev20 \
    --drones 3 4 5 6 8 --worlds 20 --out research/results/pilot-dev20
```

`research/runs/` is git-ignored (checkpoints); copy tables and CSVs into `research/results/`.

## Results so far

### G1 — headroom above Greedy (50 development worlds, `results/headroom`)

Weighted freshness (mean over worlds). Assign − Greedy is paired on the same worlds.

| Env | N | Masked random | Greedy | Assign | Assign − Greedy | Assign wins |
|---|---:|---:|---:|---:|---:|---:|
| 40×40 | 3 | 0.088 | 0.222 | 0.254 | +0.033 | 48/50 |
| 40×40 | 4 | 0.106 | 0.238 | 0.287 | +0.049 | 48/50 |
| 40×40 | 5 | 0.117 | 0.273 | 0.322 | +0.049 | 42/50 |
| 40×40 | 6 | 0.131 | 0.334 | 0.347 | +0.013 | 38/50 |
| 40×40 | 8 | 0.159 | 0.385 | 0.368 | −0.016 | 22/50 |
| 20×20 | 3 | 0.151 | 0.327 | 0.349 | +0.023 | 43/50 |
| 20×20 | 8 | 0.270 | 0.513 | 0.499 | −0.014 | 24/50 |

One-step Greedy is clearly not optimal at 3–5 drones, so a learned controller has room to
beat it. At 6–8 drones both heuristics spend hundreds of drone-steps per mission queuing for
the single charger (`charger_wait_steps` in `summary.json`); neither plans around it, which
is the interaction the structured model's charger term targets. For N > 5, Greedy is exact
within groups of up to five interacting drones and uses coordinate ascent on larger groups.
