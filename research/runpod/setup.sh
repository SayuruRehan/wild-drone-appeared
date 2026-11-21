#!/usr/bin/env bash
# One-time setup on a RunPod pod (PyTorch template). Safe to re-run.
#   bash research/runpod/setup.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "== machine"
echo "vCPUs: $(nproc)   RAM: $(free -g | awk '/Mem:/{print $2}') GB"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU visible"
python3 --version

echo "== python packages"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
    || pip install -q torch
# The repository's pyproject asks for Python >= 3.11; the code runs on 3.10+, so install the
# dependencies directly and use PYTHONPATH instead of 'pip install -e .'.
pip install -q "numpy>=1.26,<3" "gymnasium>=1.1,<2" "matplotlib>=3.8,<4" "pillow>=10,<13" pytest

echo "== tests"
PYTHONPATH=research:src python3 -m pytest -q research/tests

echo "== code version"
git log --oneline -1
echo
echo "Setup OK. Next: bash research/runpod/launch.sh"
