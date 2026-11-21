#!/usr/bin/env bash
# Launch a training matrix in the background on a RunPod pod. Survives closing the browser.
# Re-running the same command resumes: finished runs are skipped, interrupted runs continue
# from their last checkpoint (every 25k steps).
#
#   bash research/runpod/launch.sh                      # round 2 defaults (below)
#   EXPERIMENT=main40-r2 SEEDS="0 1 2" bash research/runpod/launch.sh
#
# Settings (environment variables, all optional):
#   EXPERIMENT  run-folder name                     (default main40-r2)
#   ENV_NAME    dev20 | main40                      (default main40)
#   KINDS       model kinds                         (default the round-2 candidates)
#   SEEDS       training seeds                      (default "0 1")
#   STEPS       environment steps per run           (default 300000)
#   PARALLEL    runs at once                        (default: from vCPUs and RAM)
#   RUNS_ROOT   where runs are written              (default /workspace/wild-drone-runs)
set -euo pipefail
cd "$(dirname "$0")/../.."

EXPERIMENT=${EXPERIMENT:-main40-r2}
ENV_NAME=${ENV_NAME:-main40}
KINDS=${KINDS:-"qmix qmix-overlap qmix-structured structured-overlap vdn"}
SEEDS=${SEEDS:-"0 1"}
STEPS=${STEPS:-300000}
RUNS_ROOT=${RUNS_ROOT:-/workspace/wild-drone-runs}

if [ -z "${PARALLEL:-}" ]; then
    cpus=$(nproc)
    mem_gb=$(free -g | awk '/Mem:/{print $2}')
    by_cpu=$(( cpus > 1 ? cpus - 1 : 1 ))     # leave one core for the system
    by_mem=$(( mem_gb / 4 ))                  # ~3 GB per 40x40 run incl. replay + CUDA
    PARALLEL=$(( by_cpu < by_mem ? by_cpu : by_mem ))
    [ "$PARALLEL" -lt 1 ] && PARALLEL=1
fi

mkdir -p "$RUNS_ROOT/$EXPERIMENT"
if pgrep -f "scripts/pilot.py.*$RUNS_ROOT/$EXPERIMENT" > /dev/null; then
    echo "A launcher for $EXPERIMENT is already running; not starting a second one."
    exit 1
fi

echo "experiment=$EXPERIMENT env=$ENV_NAME steps=$STEPS parallel=$PARALLEL"
echo "kinds=$KINDS"
echo "seeds=$SEEDS"
echo "code=$(git log --oneline -1)"
git log --oneline -1 > "$RUNS_ROOT/$EXPERIMENT/CODE_VERSION"

# shellcheck disable=SC2086
nohup python3 research/scripts/pilot.py --kinds $KINDS --seeds $SEEDS --steps "$STEPS" \
    --env "$ENV_NAME" --train-drones 3 --parallel "$PARALLEL" \
    --root "$RUNS_ROOT/$EXPERIMENT" -- \
    --eval-every 25000 --eval-worlds 10 --eval-drones 3 5 --device auto \
    >> "$RUNS_ROOT/$EXPERIMENT/launcher.log" 2>&1 &

sleep 5
echo
cat "$RUNS_ROOT/$EXPERIMENT/launcher.log"
echo
echo "Running in the background. Check progress with:"
echo "  python3 research/scripts/status.py $RUNS_ROOT/$EXPERIMENT"
