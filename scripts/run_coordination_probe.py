"""Run the frozen, one-seed 20x20 coordination pilot; never tunes or scales automatically."""

import argparse
import hashlib
import math
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from wild_drone.baselines import make_policy
from wild_drone.config import EnvConfig
from wild_drone.experiments import run_episode, write_rows
from wild_drone.learning import TrainConfig
from wild_drone.masked_learning import MaskedRandomPolicy, load_masked_policy, train_masked
from wild_drone.training_io import atomic_json_save

ROOT = Path(__file__).resolve().parents[1]
SCREEN = tuple(range(5700, 5720))
POLICIES = ('shaped double dqn', 'unshaped double dqn', 'masked random', 'greedy')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def screen_gate(rows):
    """Reject incomplete/duplicate evidence; gate only on original mission metrics."""
    expected = {(p, s) for p in POLICIES for s in SCREEN}
    keys = [(r['policy'], r['scenario_seed']) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('screen must contain exactly one complete mission per policy/world')
    for r in rows:
        if any(not math.isfinite(r[k]) for k in ('return', 'weighted_freshness', 'total_losses')):
            raise ValueError('screen has nonfinite metrics')
        if r['training_seed'] != (0 if 'double' in r['policy'] else None):
            raise ValueError('screen training seed differs from the frozen pilot')
    indexed = {(r['policy'], r['scenario_seed']): r for r in rows}
    pairs, comparisons = [], {}
    for reference in ('unshaped double dqn', 'masked random'):
        current = []
        for seed in SCREEN:
            candidate, base = indexed[POLICIES[0], seed], indexed[reference, seed]
            current.append({'reference': reference, 'scenario_seed': seed,
                            'return_gain': candidate['return']-base['return'],
                            'freshness_gain': candidate['weighted_freshness']-
                            base['weighted_freshness']})
        pairs.extend(current)
        gain = float(np.mean([r['return_gain'] for r in current]))
        freshness = float(np.mean([r['freshness_gain'] for r in current]))
        comparisons[reference] = {'return_gain': gain, 'freshness_gain': freshness,
                                  'passed': gain >= .5 and freshness >= .01}
    losses = sum(indexed[POLICIES[0], s]['total_losses'] for s in SCREEN)
    return {'thresholds': {'return_gain': .5, 'freshness_gain': .01,
                           'candidate_losses': 0}, 'comparisons': comparisons,
            'candidate_losses': losses,
            'passed': losses == 0 and all(c['passed'] for c in comparisons.values())}, pairs


def fit(job):
    config, settings, out = job
    train_masked('dqn', config, settings, out)
    return str(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-revision', required=True,
                        help='published source commit verified by the experiment operator')
    args = parser.parse_args(argv)
    if not re.fullmatch('[0-9a-f]{40}', args.source_revision):
        raise ValueError('source-revision must be a full Git commit SHA')
    # Refuse overwrite or cache reuse: these are prospective pilot fits.
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = EnvConfig(grid_size=20, horizon=160, battery_capacity=80, recharge_rate=5,
                       sensor_radius=1, inspection_deadline=25, max_fuel=100,
                       initial_fires=3, spread_probability=.04)
    jobs, configs = [], {}
    for recipe, weight in [('unshaped', 0), ('shaped', 10)]:
        settings = TrainConfig(episodes=100, seed=0, learning_rate=.001, double_dqn=True,
                               coordination_shaping_weight=weight, validation_interval=100,
                               validation_seeds=tuple(range(5000, 5005)), progress_interval=25)
        configs[recipe] = {'environment': config.to_dict(), 'training': settings.to_dict()}
        atomic_json_save(configs[recipe], args.out/f'{recipe}.json')
        jobs.append((config, settings, args.out/'raw'/recipe/'seed-0'))
    source_paths = sorted((ROOT/'src').rglob('*.py')) + [Path(__file__).resolve(),
                   ROOT/'docs'/'coordination-pilot-protocol.md']
    manifest = {'started_utc': datetime.now(timezone.utc).isoformat(),
                'source_revision': args.source_revision,
                'source_revision_kind': 'operator-verified published source snapshot',
                'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
                'configs': configs, 'screening_seeds': list(SCREEN),
                'checkpoint_choice': 'final', 'cpu_threads_per_process': 1,
                'fit_processes': 2}
    atomic_json_save(manifest, args.out/'launch.json')
    with ProcessPoolExecutor(max_workers=2) as pool:
        for out in pool.map(fit, jobs):
            print(f'Finished fit: {out}', flush=True)
    rows, checkpoints = [], {}
    for recipe in ('unshaped', 'shaped'):
        folder = args.out/'raw'/recipe/'seed-0'
        checkpoints[recipe] = {name: {'sha256': sha256(folder/name),
                                      'bytes': (folder/name).stat().st_size}
                               for name in ('policy.pt', 'best_policy.pt')}
        policy = load_masked_policy(folder/'policy.pt')
        rows.extend(run_episode(config, policy, s, name=f'{recipe} double dqn', training_seed=0)
                    for s in SCREEN)
    for name, make in [('masked random', MaskedRandomPolicy),
                       ('greedy', lambda s: make_policy('greedy'))]:
        rows.extend(run_episode(config, make(s), s, name=name) for s in SCREEN)
    write_rows(args.out/'screening.csv', rows)
    gate, pairs = screen_gate(rows)
    write_rows(args.out/'paired.csv', pairs)
    atomic_json_save(gate, args.out/'gate.json')
    atomic_json_save(checkpoints, args.out/'checkpoints.json')
    summary = []
    for policy in POLICIES:
        selected = [r for r in rows if r['policy'] == policy]
        summary.append({'policy': policy, 'missions': len(selected),
                        'mean_return': float(np.mean([r['return'] for r in selected])),
                        'return_sd_across_scenarios': float(np.std(
                            [r['return'] for r in selected], ddof=1)),
                        'mean_freshness': float(np.mean(
                            [r['weighted_freshness'] for r in selected])),
                        'total_losses': sum(r['total_losses'] for r in selected)})
    write_rows(args.out/'summary.csv', summary)
    print(summary, flush=True)
    print(f'Pilot gate passed: {gate["passed"]}', flush=True)


if __name__ == '__main__':
    main()
