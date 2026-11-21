# Running the study on RunPod

The simulator runs on the CPU and only the network updates use the GPU, so on a pod the
**number of vCPUs decides how many runs train in parallel** (one run ≈ one core, ~3 GB RAM).
A 40×40 run of 300k steps takes about 1.5–2 h on a dedicated core.

## 1. Create the pod

- GPU: A40 (any CUDA GPU works; the GPU is not the bottleneck).
- Template: a **RunPod PyTorch** template (Python ≥ 3.10, CUDA PyTorch preinstalled).
- Prefer a pod with **more vCPUs** (≥ 8). RAM ≥ 4 GB per parallel run.
- Volume mounted at `/workspace`, ≥ 20 GB (runs, checkpoints and replay buffers live there;
  it survives stopping the pod, not terminating it).

## 2. Get the code and set up (web terminal)

```bash
cd /workspace && git clone -b for-publishing https://github.com/SayuruRehan/wild-drone-appeared
```

```bash
cd /workspace/wild-drone-appeared && bash research/runpod/setup.sh
```

Setup prints vCPUs / RAM / GPU, installs the few dependencies, runs the test suite (must pass)
and prints the code version.

## 3. Launch (runs in the background; you can close the browser)

```bash
cd /workspace/wild-drone-appeared && bash research/runpod/launch.sh
```

Defaults: experiment `main40-r2`, 40×40, models
`qmix qmix-overlap qmix-structured structured-overlap vdn`, seeds `0 1` (10 runs),
300k steps, parallel runs chosen from vCPUs and RAM. Override with variables, e.g.
`SEEDS="0 1 2" bash research/runpod/launch.sh`.

## 4. Check progress (any time)

```bash
cd /workspace/wild-drone-appeared && python3 research/scripts/status.py /workspace/wild-drone-runs/main40-r2
```

Each run logs development freshness every 25k steps (~10–15 min). `losses` and `guard` must
stay 0. If the pod restarts, run step 3 again: finished runs are skipped and interrupted
runs resume from their last checkpoint.

## 5. When every run shows `done`: evaluate and pack the results

```bash
cd /workspace/wild-drone-appeared && bash research/runpod/evaluate.sh
```

This evaluates every run on 20 development worlds at 3, 4, 5, 6 and 8 drones, writes
`table.md` and `stats.md`, and packs the small result files (no checkpoints) into
`/workspace/wild-drone-runs/main40-r2-results.tar.gz`. Download that file and send it to
Claude to be committed with the code that produced it. Keep the run folders on the volume
until the results are committed (their `snapshots/` are needed to explain any anomaly).

Never pass `--final` (final-test worlds) until the protocol is frozen.
