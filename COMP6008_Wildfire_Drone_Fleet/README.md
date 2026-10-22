# Wildfire Drone Fleet — COMP6008 Reinforcement Learning

A central controller commands three battery-limited drones that repeatedly inspect a
simulated wildfire and share one charger. The task is modelled as a fully observable MDP and
solved with three RL agents — **DQN**, **REINFORCE with a learned baseline** and
**synchronous A2C** — which are compared with conventional controllers.

The project notebook is **`Wildfire_Drone_Fleet_clean.ipynb`**. It contains the MDP,
environment, agents, pseudocode, experiments, tuning, results and critical review.

## How to run the notebook

You need **Python 3.11 or 3.12** (check with `python3 --version`). A GPU is not needed.

1. Unzip `COMP6008_Wildfire_Drone_Fleet.zip` and open a terminal in the unzipped folder:

   ```bash
   cd COMP6008_Wildfire_Drone_Fleet
   ```

2. Create and activate a virtual environment:

   ```bash
   # macOS / Linux
   python3.12 -m venv .venv
   source .venv/bin/activate

   # Windows
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   ```

3. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Open the notebook and run it:

   ```bash
   python -m jupyter notebook Wildfire_Drone_Fleet_clean.ipynb
   ```

   In Jupyter choose **Kernel → Restart Kernel and Run All Cells**. It takes about 2 minutes on a laptop
   CPU and should finish without errors.

By default the notebook reloads the archived results of our experiments and only trains small
demonstration models. Full retraining is optional and slow; it is switched on with the
`RUN_...` flags at the top of Section 1.

## Optional: run the tests

With the virtual environment still active:

```bash
python -m pip install -e .
python -m pytest -q
```

Expected result: `129 passed, 5 skipped`.

## Common errors

| Error | Fix |
| --- | --- |
| `No matching distribution found for torch==2.5.1` | Your Python version is not 3.11 or 3.12 (for example 3.13). Install Python 3.12 and repeat steps 2–3. |
| `ModuleNotFoundError: No module named 'torch'` (or `gymnasium`, `pandas`) in the notebook | Jupyter is using a different Python. Activate the virtual environment (step 2) and start Jupyter from it (step 4). |
| `FileNotFoundError: Cannot find COMP6008_Wildfire_Drone_Fleet/configs` | The notebook was moved out of its folder. Keep the unzipped folder intact and open the notebook from inside it. |
| `.venv\Scripts\activate` is blocked on Windows PowerShell | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt instead. |

## What is in this folder

| Path | Contents |
| --- | --- |
| `Wildfire_Drone_Fleet_clean.ipynb` | The project notebook (all code it runs is defined in it) |
| `requirements.txt` | Pinned Python dependencies |
| `configs/` | Environment and training settings |
| `docs/experiments/` | Archived results the notebook reads: 20×20 study, 40×40 main study, 40×40 follow-ups |
| `src/wild_drone/`, `scripts/`, `tests/`, `pyproject.toml` | Package version of the code that produced the archived results, with its tests |
| `*_DeclarationOfOriginality.pdf` | Signed declarations of the three group members |

The fire, wind and energy settings are illustrative simulation values, not calibrated
real-world physics.
