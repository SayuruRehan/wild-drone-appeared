"""Launch a matrix of training runs in parallel (one CPU thread each, or one GPU).

    PYTHONPATH=research:src python research/scripts/pilot.py --kinds joint vdn dcg structured \
        --seeds 0 1 --steps 60000 --env dev20 --parallel 4 --root research/runs/pilot-dev20

Extra arguments after ``--`` are passed to ``fleetrl.train`` unchanged.
A run whose ``done.json`` exists is skipped, so the script can be re-launched after an
interruption (e.g. a Colab disconnect).
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kinds', nargs='+', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1])
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--env', default='dev20')
    p.add_argument('--train-drones', nargs='+', default=['3'])
    p.add_argument('--parallel', type=int, default=4)
    p.add_argument('--root', required=True)
    a, extra = p.parse_known_args()
    if extra and extra[0] == '--':
        extra = extra[1:]
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=f"{HERE}:{HERE.parent/'src'}", OMP_NUM_THREADS='1')
    queue = []
    for seed in a.seeds:
        for kind in a.kinds:
            out = root/f'{kind}-{seed}'
            if (out/'done.json').exists():
                print('skip', out, flush=True)
                continue
            cmd = [sys.executable, '-m', 'fleetrl.train', '--kind', kind, '--seed', str(seed),
                   '--steps', str(a.steps), '--env', a.env, '--train-drones', *a.train_drones,
                   '--out', str(out), *extra]
            queue.append((out, cmd))
    running = []
    while queue or running:
        while queue and len(running) < a.parallel:
            out, cmd = queue.pop(0)
            out.mkdir(parents=True, exist_ok=True)
            log = open(out/'stdout.txt', 'a')
            running.append((out, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                                  env=env)))
            print('start', out, flush=True)
        for item in list(running):
            out, proc = item
            if proc.poll() is not None:
                running.remove(item)
                print('done' if proc.returncode == 0 else f'FAILED ({proc.returncode})', out,
                      flush=True)
        time.sleep(5)


if __name__ == '__main__':
    main()
