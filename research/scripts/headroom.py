"""Gate 1: is there headroom above one-step Greedy, and how does it change with fleet size?

Flies the non-learning controllers on development worlds only (never final-test worlds)
and writes per-episode rows plus a summary.

    PYTHONPATH=research:src python research/scripts/headroom.py --env dev20 --drones 3 5 8 \
        --policies greedy assign masked_random --worlds 20 --workers 4 --out research/results/headroom
"""

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent/'src')]

from fleetrl.baselines import make_baseline  # noqa: E402
from fleetrl.evaluate import run_episode  # noqa: E402
from wild_drone.config import EnvConfig  # noqa: E402


def job(args):
    env_name, n, policy, seed = args
    config = EnvConfig(**json.load(open(HERE/'configs/environments.json'))[env_name])
    row = run_episode(config, n, make_baseline(policy, seed), seed)
    return {'env': env_name, 'policy': policy, **row}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', nargs='+', default=['dev20'])
    p.add_argument('--drones', nargs='+', type=int, default=[3, 5, 8])
    p.add_argument('--policies', nargs='+', default=['greedy', 'assign', 'masked_random'])
    p.add_argument('--worlds', type=int, default=20)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out', default=str(HERE/'results/headroom'))
    a = p.parse_args()
    split = json.load(open(HERE/'configs/splits.json'))['development']
    seeds = list(range(split['start'], split['start']+a.worlds))
    assert seeds[-1] < split['stop'], 'not enough development worlds'
    jobs = [(e, n, pol, s) for e in a.env for n in a.drones for pol in a.policies for s in seeds]
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(job, jobs, chunksize=1))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with open(out/'episodes.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    metrics = ['weighted_freshness', 'overdue_fraction', 'return', 'losses',
               'duplicate_cells_per_step', 'charger_wait_steps', 'decision_ms']
    summary = []
    for e in a.env:
        for n in a.drones:
            greedy = {r['seed']: r['weighted_freshness'] for r in rows
                      if r['env'] == e and r['n_drones'] == n and r['policy'] == 'greedy'}
            for pol in a.policies:
                sel = [r for r in rows if r['env'] == e and r['n_drones'] == n
                       and r['policy'] == pol]
                entry = {'env': e, 'n_drones': n, 'policy': pol, 'worlds': len(sel)}
                for m in metrics:
                    v = np.array([r[m] for r in sel], dtype=float)
                    entry[m] = float(v.mean())
                    entry[m+'_sd'] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
                if greedy:
                    diff = np.array([r['weighted_freshness']-greedy[r['seed']] for r in sel])
                    entry['vs_greedy_mean'] = float(diff.mean())
                    entry['vs_greedy_wins'] = int((diff > 0).sum())
                summary.append(entry)
    with open(out/'summary.json', 'w') as f:
        json.dump({'seeds': seeds, 'summary': summary}, f, indent=1)
    print(f"{'env':6} {'N':>2} {'policy':14} {'fresh':>7} {'sd':>6} {'Δgreedy':>8} "
          f"{'wins':>4} {'dup/step':>8} {'wait':>6} {'ms':>6}")
    for s in summary:
        print(f"{s['env']:6} {s['n_drones']:>2} {s['policy']:14} {s['weighted_freshness']:7.4f} "
              f"{s['weighted_freshness_sd']:6.4f} {s.get('vs_greedy_mean', 0):+8.4f} "
              f"{s.get('vs_greedy_wins', 0):>4} {s['duplicate_cells_per_step']:8.2f} "
              f"{s['charger_wait_steps']:6.1f} {s['decision_ms']:6.1f}")


if __name__ == '__main__':
    main()
