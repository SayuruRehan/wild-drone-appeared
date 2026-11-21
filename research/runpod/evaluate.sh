#!/usr/bin/env bash
# After training: evaluate every finished run on 20 development worlds at N = 3,4,5,6,8,
# compute the statistics, and pack the small result files for download.
#
#   bash research/runpod/evaluate.sh                    # EXPERIMENT=main40-r2
#   EXPERIMENT=main40-r2 WORKERS=8 bash research/runpod/evaluate.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

EXPERIMENT=${EXPERIMENT:-main40-r2}
ENV_NAME=${ENV_NAME:-main40}
RUNS_ROOT=${RUNS_ROOT:-/workspace/wild-drone-runs}
WORLDS=${WORLDS:-20}
WORKERS=${WORKERS:-$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))}
SRC="$RUNS_ROOT/$EXPERIMENT"
OUT="$RUNS_ROOT/$EXPERIMENT-eval"
export PYTHONPATH=research:src

if pgrep -f "fleetrl.train.*$SRC" > /dev/null; then
    echo "Training for $EXPERIMENT is still running; evaluating only finished runs would mix"
    echo "checkpoints. Wait until status.py shows every run as done."
    exit 1
fi

python3 research/scripts/evaluate_runs.py --root "$SRC" --drones 3 4 5 6 8 --worlds "$WORLDS" \
    --workers "$WORKERS" --out "$OUT"
python3 research/scripts/stats.py --episodes "$OUT/episodes.csv" \
    --baselines research/results/headroom/episodes.csv --env "$ENV_NAME" \
    --compare qmix-structured:qmix qmix-overlap:qmix structured-overlap:vdn qmix:vdn \
    --out "$OUT/stats.md"

# small files only (no checkpoints): configs, learning curves, tables, per-episode CSV
PACK="$RUNS_ROOT/$EXPERIMENT-results.tar.gz"
tar -czf "$PACK" -C "$RUNS_ROOT" \
    "$EXPERIMENT-eval" \
    $(cd "$RUNS_ROOT" && find "$EXPERIMENT" -maxdepth 2 \
        \( -name config.json -o -name log.jsonl -o -name done.json -o -name CODE_VERSION \
           -o -name launcher.log \) -print)
echo
cat "$OUT/table.md"
echo
echo "Packed results: $PACK ($(du -h "$PACK" | cut -f1))"
echo "Download it (RunPod file browser / Jupyter, or scp) and send it to Claude."
