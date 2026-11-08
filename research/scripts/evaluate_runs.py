"""Evaluate trained runs on development worlds across fleet sizes, next to the baselines.

    PYTHONPATH=research:src python research/scripts/evaluate_runs.py \
        --root research/runs/pilot-dev20 --drones 3 4 5 6 8 --worlds 20 --workers 4 \
        --baselines research/results/headroom/episodes.csv --out research/results/pilot-dev20

Learner statistics follow the frozen-study convention: mean over worlds within each
training seed, then mean and SD across seeds. Paired differences against Greedy use the
same worlds. Final-test worlds are refused unless ``--final`` is given.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent/'src')]


def job(args):
    run_dir, checkpoint, n, seed = args
    import torch
    torch.set_num_threads(1)
    from fleetrl.evaluate import run_episode
    from fleetrl.train import load_policy
    policy, cfg, config = load_policy(run_dir, checkpoint)
    row = run_episode(config, n, policy, seed)
    return {'policy': cfg.kind, 'train_seed': cfg.seed, 'env': cfg.env, **row}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--checkpoint', default='final.pt')
    p.add_argument('--drones', nargs='+', type=int, default=[3, 4, 5, 6, 8])
    p.add_argument('--worlds', type=int, default=20)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--baselines', default=str(HERE/'results/headroom/episodes.csv'))
    p.add_argument('--out', required=True)
    p.add_argument('--final', action='store_true', help='evaluate on final-test worlds')
    a = p.parse_args()
    split = json.load(open(HERE/'configs/splits.json'))['final_test' if a.final else 'development']
    seeds = list(range(split['start'], split['start']+a.worlds))
    runs = sorted(d for d in Path(a.root).iterdir() if (d/a.checkpoint).exists())
    jobs = []
    for d in runs:
        cfg = json.load(open(d/'config.json'))
        sizes = cfg['train_drones'] if cfg['kind'] == 'joint' else a.drones
        jobs += [(str(d), a.checkpoint, n, s) for n in sizes if n in a.drones for s in seeds]
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(job, jobs, chunksize=1))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with open(out/'episodes.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)

    env_name = rows[0]['env']
    base = defaultdict(dict)
    if a.baselines and Path(a.baselines).exists() and not a.final:
        for r in csv.DictReader(open(a.baselines)):
            if r['env'] == env_name and int(r['seed']) in seeds:
                base[(r['policy'], int(r['n_drones']))][int(r['seed'])] = r
    table = []
    for n in a.drones:
        greedy = {s: float(r['weighted_freshness']) for s, r in base[('greedy', n)].items()}
        for pol in ('masked_random', 'greedy', 'assign'):
            sel = base[(pol, n)]
            if len(sel) == len(seeds):
                v = np.array([float(sel[s]['weighted_freshness']) for s in seeds])
                d = v - np.array([greedy[s] for s in seeds]) if greedy else v*0
                table.append({'n': n, 'policy': pol, 'fresh': v.mean(), 'sd': v.std(ddof=1),
                              'seeds': '-', 'vs_greedy': d.mean(), 'wins': int((d > 0).sum()),
                              'dup': np.mean([float(sel[s]['duplicate_cells_per_step'])
                                              for s in seeds]),
                              'wait': np.mean([float(sel[s]['charger_wait_steps'])
                                               for s in seeds]), 'losses': 0})
        for kind in sorted({r['policy'] for r in rows}):
            sel = [r for r in rows if r['policy'] == kind and r['n_drones'] == n]
            if not sel:
                continue
            by_seed = defaultdict(list)
            for r in sel:
                by_seed[r['train_seed']].append(r['weighted_freshness'])
            means = np.array([np.mean(v) for v in by_seed.values()])
            diffs = [r['weighted_freshness']-greedy[r['seed']] for r in sel if r['seed'] in greedy]
            table.append({'n': n, 'policy': kind, 'fresh': means.mean(),
                          'sd': means.std(ddof=1) if len(means) > 1 else 0.0,
                          'seeds': len(means), 'vs_greedy': np.mean(diffs) if diffs else 0.0,
                          'wins': f"{sum(x > 0 for x in diffs)}/{len(diffs)}",
                          'dup': np.mean([r['duplicate_cells_per_step'] for r in sel]),
                          'wait': np.mean([r['charger_wait_steps'] for r in sel]),
                          'losses': int(sum(r['losses'] for r in sel))})
    lines = ['| N | policy | freshness | SD | train seeds | Δ vs Greedy | wins | dup/step | '
             'charger wait | losses |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for t in table:
        lines.append(f"| {t['n']} | {t['policy']} | {t['fresh']:.4f} | {t['sd']:.4f} | "
                     f"{t['seeds']} | {t['vs_greedy']:+.4f} | {t['wins']} | {t['dup']:.2f} | "
                     f"{t['wait']:.1f} | {t['losses']} |")
    text = '\n'.join(lines)
    (out/'table.md').write_text(f'Worlds: {seeds[0]}-{seeds[-1]} ({env_name}), checkpoint '
                                f'{a.checkpoint}\n\n{text}\n')
    print(text)


if __name__ == '__main__':
    main()
