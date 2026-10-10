"""Tuning, held-out test and sensitivity studies for the masked learners.

Three stages, each resumable (a finished fit is skipped on rerun):

  tune         one-factor sweeps on validation scenarios 5000-5004 only
  test         frozen settings, 3 training seeds, final test scenarios 10000-10019
  sensitivity  battery capacity, recharge rate and fire spread varied one at a time

Test scenarios are never used for any choice: `test` reads the frozen settings from
`--settings` (written by hand after inspecting the `tune` output).
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from wild_drone.baselines import make_policy
from wild_drone.config import EnvConfig
from wild_drone.experiments import plot_learning, run_episode, validate_seed_split, write_rows
from wild_drone.learning import TrainConfig, load_policy, train
from wild_drone.masked_learning import MaskedRandomPolicy, load_masked_policy, train_masked
from wild_drone.study_cache import validate_cached_fit

VALIDATION = list(range(5000, 5005))
TEST = list(range(10000, 10020))
PLANNER = dict(horizon=3, candidates=8, samples=2)

# One factor at a time around the defaults; the other settings stay at their defaults.
TUNING_GRID = {
    'dqn': {'learning_rate': [0.0001, 0.0003, 0.001]},
    'reinforce': {'learning_rate': [0.0001, 0.0003, 0.001], 'entropy_weight': [0.0, 0.01, 0.05]},
    'a2c': {'learning_rate': [0.0001, 0.0003, 0.001], 'entropy_weight': [0.0, 0.01, 0.05],
            'n_steps': [8, 16, 32]},
}
SENSITIVITY = {'battery_capacity': [60.0, 120.0], 'recharge_rate': [3.0, 8.0],
               'spread_probability': [0.02, 0.08]}


def fit(job):
    """Train one learner or reuse a complete, matching fit."""
    masked, algorithm, config, settings, out = job
    validate_study_split(settings)
    if not validate_cached_fit(masked, algorithm, config, settings, out):
        if masked:
            train_masked(algorithm, config, settings, out)
        else:
            train(algorithm, config, settings, out)
    return str(out)


def settings_for(raw, seed, episodes, **overrides):
    return TrainConfig(**(raw.get('training', {}) | {
        'episodes': episodes, 'seed': seed, 'device': 'cpu', 'validation_seeds': VALIDATION,
        'validation_interval': 25, 'checkpoint_interval': 25, 'progress_interval': 50}
        | overrides))


def run_jobs(jobs, workers):
    # Fail before starting any worker, including when an earlier job is entirely fresh.
    for masked, algorithm, config, settings, out in jobs:
        validate_study_split(settings)
        validate_cached_fit(masked, algorithm, config, settings, out)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for out in pool.map(fit, jobs):
            print(f'finished {out}', flush=True)


def validate_study_split(settings):
    validate_seed_split(settings.scenario_seed_start, settings.episodes, settings.validation_seeds)
    validate_seed_split(settings.scenario_seed_start, settings.episodes, TEST)
    if set(settings.validation_seeds) & set(TEST):
        raise ValueError('validation and final test scenario seeds overlap')


def baseline_rows(config, seeds, planner=True):
    makers = {'random': lambda s: make_policy('random', s), 'masked random': MaskedRandomPolicy,
              'all stay': lambda s: StayPolicy(), 'sweep': lambda s: make_policy('sweep'),
              'greedy': lambda s: make_policy('greedy')}
    if planner:
        makers['planner'] = lambda s: make_policy('planner', s, **PLANNER)
    return [run_episode(config, make(s), s, name=name)
            for name, make in makers.items() for s in seeds]


class StayPolicy:
    """Every drone stays at base: the 'do nothing' reference."""

    def act(self, env, observation):
        return 0


def summarise(rows, path):
    """Mean over scenarios per fit, then mean and SD over fits (baselines: over scenarios)."""
    out = []
    for name in dict.fromkeys(r['policy'] for r in rows):
        selected = [r for r in rows if r['policy'] == name]
        seeds = list(dict.fromkeys(r['training_seed'] for r in selected))
        groups = ([[r] for r in selected] if seeds == [None] else
                  [[r for r in selected if r['training_seed'] == s] for s in seeds])

        def unit(metric):
            return [float(np.mean([r[metric] for r in group])) for group in groups]
        returns = unit('return')
        out.append({'policy': name, 'return_mean': np.mean(returns),
                    'return_sd': np.std(returns, ddof=1) if len(returns) > 1 else 0.0,
                    **{f'{m}_mean': np.mean(unit(m)) for m in
                       ['weighted_freshness', 'overdue_fraction', 'energy_used', 'total_losses',
                        'safe_returns', 'charge_completions', 'collision_attempts',
                        'decision_ms']}})
    write_rows(path, out)
    for row in out:
        print(f"{row['policy']:32s} {row['return_mean']:+8.3f} ± {row['return_sd']:6.3f}  "
              f"fresh {row['weighted_freshness_mean']:.3f}  "
              f"losses {row['total_losses_mean']:.2f}", flush=True)
    return out


def tune(args, raw, config):
    jobs = []
    for algorithm, grid in TUNING_GRID.items():
        for name, values in grid.items():
            for value in values:
                for seed in args.seeds:
                    out = args.out/'tune'/algorithm/f'{name}={value}'/f'seed-{seed}'
                    jobs.append((True, algorithm, config,
                                 settings_for(raw, seed, args.episodes, **{name: value}), out))
    run_jobs(jobs, args.jobs)
    rows = []
    for _, algorithm, _, settings, out in jobs:
        summary = json.loads((out/'training_summary.json').read_text())
        label = out.parent.name
        name, value = label.split('=')
        validation = [r for r in csv_rows(out/'validation.csv')
                      if int(r['episode']) == settings.episodes]
        rows.append({'algorithm': algorithm, 'parameter': name, 'value': value,
                     'seed': settings.seed,
                     'final_validation_return': np.mean([float(r['return']) for r in validation]),
                     'final_validation_freshness': np.mean([float(r['weighted_freshness'])
                                                            for r in validation]),
                     'final_validation_losses': np.mean([float(r['total_losses'])
                                                         for r in validation]),
                     'best_validation_return': summary['best_validation_return'],
                     'last100_training_return': np.mean(
                         [float(r['return']) for r in csv_rows(out/'training.csv')][-100:])})
    write_rows(args.out/'tune'/'tuning.csv', rows)
    for row in rows:
        print(f"{row['algorithm']:9s} {row['parameter']:15s} {row['value']:>7s} "
              f"seed {row['seed']}  final val {row['final_validation_return']:+8.3f}", flush=True)


def csv_rows(path):
    import csv
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def test(args, raw, config):
    frozen = json.loads(args.settings.read_text())
    jobs = []
    for masked in (True, False):
        for algorithm, overrides in frozen.items():
            for seed in args.seeds:
                kind = 'masked' if masked else 'unmasked'
                out = args.out/'test'/kind/algorithm/f'seed-{seed}'
                jobs.append((masked, algorithm, config,
                             settings_for(raw, seed, args.episodes, **overrides), out))
    run_jobs(jobs, args.jobs)
    rows = []
    for masked, algorithm, _, settings, out in jobs:
        kind = 'masked' if masked else 'unmasked'
        policy = (load_masked_policy if masked else load_policy)(out/'policy.pt')
        rows += [run_episode(config, policy, s, name=f'{kind} {algorithm}',
                             training_seed=settings.seed) for s in TEST]
    rows += baseline_rows(config, TEST)
    write_rows(args.out/'test'/'test_episodes.csv', rows)
    summarise(rows, args.out/'test'/'test_summary.csv')
    for kind in ('masked', 'unmasked'):
        for algorithm in frozen:
            plot_learning([args.out/'test'/kind/algorithm/f'seed-{s}'/'training.csv'
                           for s in args.seeds],
                          args.out/'test'/f'learning-{kind}-{algorithm}.png')


def sensitivity(args, raw, config):
    frozen = json.loads(args.settings.read_text())
    algorithm = args.algorithm
    conditions = [('default', None, None)] + [(f'{k}={v}', k, v) for k, vs in
                                               SENSITIVITY.items() for v in vs]
    jobs = []
    for label, key, value in conditions:
        condition = config if key is None else replace(config, **{key: value})
        for seed in args.seeds:
            out = args.out/'sensitivity'/label/algorithm/f'seed-{seed}'
            jobs.append((True, algorithm, condition,
                         settings_for(raw, seed, args.episodes, **frozen[algorithm]), out))
    run_jobs(jobs, args.jobs)
    rows = []
    for label, key, value in conditions:
        condition = config if key is None else replace(config, **{key: value})
        for seed in args.seeds:
            out = args.out/'sensitivity'/label/algorithm/f'seed-{seed}'
            policy = load_masked_policy(out/'policy.pt')
            rows += [dict(condition=label, **run_episode(
                condition, policy, s, name=f'masked {algorithm}', training_seed=seed))
                for s in TEST]
        rows += [dict(condition=label, **row) for row in baseline_rows(condition, TEST)]
    write_rows(args.out/'sensitivity'/'sensitivity_episodes.csv', rows)
    summary = []
    for label, *_ in conditions:
        print(f'\n== {label}', flush=True)
        for row in summarise([r for r in rows if r['condition'] == label],
                             args.out/'sensitivity'/f'summary-{label}.csv'):
            summary.append({'condition': label, **row})
    write_rows(args.out/'sensitivity'/'sensitivity_summary.csv', summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['tune', 'test', 'sensitivity'])
    parser.add_argument('--config', type=Path, default=Path('configs/development.json'))
    parser.add_argument('--settings', type=Path, default=Path('configs/frozen-settings.json'),
                        help='per-algorithm overrides chosen from the tuning stage')
    parser.add_argument('--algorithm', default='a2c', help='learner for the sensitivity stage')
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--out', type=Path, default=Path('runs/final'))
    args = parser.parse_args(argv)
    raw = json.loads(args.config.read_text())
    config = EnvConfig(**raw['environment'])
    {'tune': tune, 'test': test, 'sensitivity': sensitivity}[args.stage](args, raw, config)


if __name__ == '__main__':
    main()
