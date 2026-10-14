# Wild Drone Appeared — mission control

The custom React/TypeScript interface has two modes: a static evidence showcase and a
local FastAPI simulation lab. Both use actual simulator states. No JavaScript wildfire
or flight model approximates the Python implementation.

For a full explanation of the screens, transition rules, animation pipeline and results,
read the [team guide](mission-control-team-guide.md). The
[WhatsApp summary](mission-control-whatsapp.txt) is ready to copy and share.

## Run locally

From the repository root, use Python 3.11+ and Node 22.12+:

Activate your virtual environment first (macOS/Linux: `source .venv/bin/activate`).
Create one with `python3 -m venv .venv` if it does not exist.

```bash
python -m pip install -e '.[ui,rl]'
cd web
npm ci
npm run build
cd ..
wild-drone-ui
```

Open http://127.0.0.1:8000. The default service binds to localhost. To select a port,
use `wild-drone-ui --port 8001`. `python -m wild_drone.webapp` is equivalent.

For frontend development, run the Python service and `npm run dev` in `web/` in
separate terminals. Vite proxies `/api` to port 8000. A static `npm run preview`
shows the replay experience without a connected simulation lab.

### Frozen learners

The five baseline controllers are available without checkpoint files. Live RL requires
the nine original frozen fits downloaded using the archive's GitHub Actions instructions.
Their expected layout is `<fits>/dqn/seed-0/policy.pt`, and similarly for all methods/seeds.

```bash
wild-drone-ui --checkpoint-root runs/rl-main-2026-10-11/fits
```

Every checkpoint must match its archived SHA-256, algorithm, fitted seed and full environment
and training configuration. Missing or different files are unavailable, never replaced with
a random model or a historical checkpoint. Live learners require the original 40×40
environment. Baselines may explore the sandbox's bounded battery, charging and spread ranges.

## Playback and evidence

The interface bundles worlds 22000, 22009 and 22019 for nine fitted learners and four baselines:
39 recorded missions. Archived actions reconstruct fire, fuel, inspection age, drone states,
charging and wind. The exporter checks every pre/post state hash, reward component, mask,
termination, and full-horizon summary. Archived neural scores are displayed as Q estimates
or logits, with their feasible actions; they are not explanations of network causality.

Replay files are gzip-compressed and loaded individually. The browser checks each compressed
file against its SHA-256 before parsing. Full-study chart means and sample SDs are independently
recomputed from all 260 mission rows, not the three selected replay worlds. Follow-up planner,
sensitivity and learning-rate results are kept distinct from the frozen comparison.

```bash
python scripts/export_web_evidence.py
```

This regenerates `web/public/data/`. It performs no training, modifies no archive, and needs
no learned-policy weights. The exporter requires the project's NumPy/Gymnasium/Torch runtime.
Inputs and hashes are recorded in `evidence.json`. Default replay controls support space to
play/pause, arrows to step, a timeline slider, cell coordinate inputs and reduced-motion mode.

## API and bounded jobs

- `GET /api/capabilities`: baseline and verified RL availability.
- `POST /api/missions`: `{policy, world, grid, battery, recharge, spread}` → job ID, HTTP 202.
- `GET /api/missions/{id}?after_step=N`: status, initial packet, frames newer than N, final
  summary or error. Initial frame is step 0; use `after_step=-1` to request all available frames.
- `DELETE /api/missions/{id}`: cooperative cancellation at a simulator step boundary.

There are two simulation workers and at most eight retained jobs. Finished jobs expire after
30 minutes and are cleaned up on the next creation; the oldest finished job may be evicted
when capacity is reached. Full queues reject new work with HTTP 429. Jobs are isolated in memory,
lost on restart, and do not launch training or overwrite committed evidence. A planner decision
already in progress finishes before cancellation. Completed sandbox runs can be downloaded.

The private hosted showcase serves static files only. It explains how to start the lab rather
than contacting visitors' local machines. Sites identity lives in `web/.openai/hosting.json`;
the static directory is `dist`. Source synchronization for publication uses a separate sanitized
checkout, preserving the GitHub repository and the user's `sula` branch.

## Checks

```bash
python -m pytest -q
python -m ruff check src tests scripts
cd web
npm test
npm run build
```

Historical results, missing frozen checkpoints, safety-mask contributions, illustrative physics,
uncertainty units and unsuccessful performance gates remain visible throughout the experience.

## Verified delivery snapshot

On 13 October 2026, all 39 exports passed their 11,389 transition checks. Python
regressions verified 252 passing tests with five CUDA-only skips: 250 passed inside
the sandbox, and the two local-HTTP/shared-memory tests passed with the required host
access. The 13 UI API/evidence cases also passed after the final backend refinements.
Ruff and the rendered smoke mission passed. Frontend type/build checks and two Node
playback tests passed; installation audited with zero reported vulnerabilities.
Browser checks covered desktop and mobile playback, synchronized seeking, freshness
layers, the DQN score inspector, responsive navigation, results and a newly generated
20×20 sandbox mission. Mobile document width matched scroll width (no horizontal overflow).
