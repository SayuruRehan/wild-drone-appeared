"""Statistics for the study tables (protocol section 6), numpy only.

    PYTHONPATH=research:src python research/scripts/stats.py \
        --episodes research/results/pilot-dev20/episodes.csv \
        --baselines research/results/headroom/episodes.csv --env dev20 \
        --reference greedy --compare structured:vdn structured:dcg --out stats.md

For each (fleet size, controller):
  * IQM of weighted freshness over all (training seed x world) scores, with a 95%
    two-level bootstrap CI (resample training seeds, then worlds; Agarwal et al., 2021);
  * paired difference against ``--reference`` on the same worlds (seed-averaged per world),
    with a bootstrap CI and a two-sided sign-flip permutation p-value;
  * the same for every ``A:B`` pair in ``--compare``.
p-values within each fleet size are Holm-corrected over all comparisons in the table.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def iqm(x):
    x = np.sort(np.asarray(x, dtype=float).ravel())
    n = len(x)
    lo, hi = int(np.floor(0.25*n)), int(np.ceil(0.75*n))
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())


def bootstrap_iqm(scores, rng, reps=2000):
    """scores: (seeds, worlds) array -> (iqm, lo, hi)."""
    s, w = scores.shape
    stats = np.empty(reps)
    for r in range(reps):
        si = rng.integers(0, s, s)
        wi = rng.integers(0, w, w)
        stats[r] = iqm(scores[np.ix_(si, wi)])
    return iqm(scores), float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def paired(diff, rng, reps=10000):
    """diff: per-world differences -> (mean, lo, hi, p) with sign-flip permutation p."""
    diff = np.asarray(diff, dtype=float)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)])
    observed = abs(diff.mean())
    flips = rng.choice([-1.0, 1.0], size=(reps, len(diff)))
    p = (np.sum(np.abs((flips*diff).mean(1)) >= observed - 1e-15) + 1)/(reps+1)
    return (float(diff.mean()), float(np.quantile(boots, 0.025)),
            float(np.quantile(boots, 0.975)), float(p))


def holm(pvalues):
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m-rank)*pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def load(episodes, baselines, env):
    """-> scores[(policy, n)][train_seed][world] = freshness"""
    scores = defaultdict(lambda: defaultdict(dict))
    for path in episodes:
        for r in csv.DictReader(open(path)):
            scores[(r['policy'], int(r['n_drones']))][r.get('train_seed', '-')][int(r['seed'])] = \
                float(r['weighted_freshness'])
    if baselines:
        for r in csv.DictReader(open(baselines)):
            if r['env'] == env:
                scores[(r['policy'], int(r['n_drones']))]['-'][int(r['seed'])] = \
                    float(r['weighted_freshness'])
    return scores


def matrix(table, worlds):
    seeds = sorted(table)
    return np.array([[table[s][w] for w in worlds] for s in seeds])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', nargs='+', required=True)
    p.add_argument('--baselines')
    p.add_argument('--env', required=True)
    p.add_argument('--reference', default='greedy')
    p.add_argument('--compare', nargs='*', default=[])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out')
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    scores = load(a.episodes, a.baselines, a.env)
    learner_worlds = {}
    for (pol, n), table in scores.items():
        if '-' not in table:
            worlds = set.intersection(*(set(t) for t in table.values()))
            learner_worlds[n] = learner_worlds.get(n, worlds) & worlds
    lines = []
    for n in sorted(learner_worlds):
        worlds = sorted(learner_worlds[n])
        lines += [f'### N = {n} ({len(worlds)} worlds)', '',
                  f'| controller | IQM [95% CI] | Δ vs {a.reference} [95% CI] | p (Holm) |',
                  '|---|---|---|---:|']
        rows, pvals = [], []
        ref = scores.get((a.reference, n))
        ref_world = matrix(ref, worlds).mean(0) if ref else None
        for (pol, k), table in sorted(scores.items()):
            if k != n or not all(all(w in t for w in worlds) for t in table.values()):
                continue
            m = matrix(table, worlds)
            est = bootstrap_iqm(m, rng)
            if ref_world is not None and pol != a.reference:
                d = paired(m.mean(0) - ref_world, rng)
                rows.append([pol, est, d])
                pvals.append(d[3])
            else:
                rows.append([pol, est, None])
        comps = []
        for spec in a.compare:
            x, y = spec.split(':')
            if (x, n) in scores and (y, n) in scores:
                d = paired(matrix(scores[(x, n)], worlds).mean(0) -
                           matrix(scores[(y, n)], worlds).mean(0), rng)
                comps.append((f'{x} − {y}', d))
                pvals.append(d[3])
        adjusted = holm(np.array(pvals)) if pvals else []
        j = 0
        for pol, est, d in rows:
            cell = '–'
            padj = ''
            if d is not None:
                cell = f'{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]'
                padj = f'{adjusted[j]:.3g}'
                j += 1
            lines.append(f'| {pol} | {est[0]:.4f} [{est[1]:.4f}, {est[2]:.4f}] | {cell} | {padj} |')
        for name, d in comps:
            lines.append(f'| *{name}* | | {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] | '
                         f'{adjusted[j]:.3g} |')
            j += 1
        lines.append('')
    text = '\n'.join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text+'\n')


if __name__ == '__main__':
    main()
