"""Masked DQN/REINFORCE/A2C versus baselines on validation scenarios (never final test seeds).

Each (algorithm, training seed) fit trains into OUT/ALGORITHM/seed-K; a finished fit is
skipped on rerun, so an interrupted study restarts only the unfinished fits.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from wild_drone.baselines import make_policy
from wild_drone.config import EnvConfig
from wild_drone.experiments import (
    plot_learning,
    plot_results,
    run_episode,
    validate_seed_split,
    write_rows,
)
from wild_drone.learning import TrainConfig
from wild_drone.masked_learning import MaskedRandomPolicy, load_masked_policy, train_masked
from wild_drone.study_cache import validate_cached_fit


def fit(job):
    algorithm, config, settings, out = job
    validate_seed_split(settings.scenario_seed_start, settings.episodes, settings.validation_seeds)
    if not validate_cached_fit(True, algorithm, config, settings, out):
        train_masked(algorithm, config, settings, out)
    return algorithm, settings.seed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/development.json'))
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--training-seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--validation-seeds', nargs='+', type=int, default=list(range(5000, 5005)))
    parser.add_argument('--algorithms', nargs='+', choices=['dqn', 'reinforce', 'a2c'],
                        default=['dqn', 'reinforce', 'a2c'])
    parser.add_argument('--validation-interval', type=int, default=25)
    parser.add_argument('--jobs', type=int, default=4, help='fits trained in parallel')
    parser.add_argument('--out', type=Path, default=Path('runs/masked-study'))
    args = parser.parse_args(argv)
    raw = json.loads(args.config.read_text())
    config = EnvConfig(**raw['environment'])
    jobs = []
    for algorithm in args.algorithms:
        for seed in args.training_seeds:
            settings = TrainConfig(**(raw.get('training', {}) | {
                'episodes': args.episodes, 'seed': seed, 'device': 'cpu',
                'validation_seeds': args.validation_seeds,
                'validation_interval': args.validation_interval, 'progress_interval': 50}))
            jobs.append((algorithm, config, settings, args.out/algorithm/f'seed-{seed}'))
    for algorithm, config, settings, out in jobs:
        validate_seed_split(settings.scenario_seed_start, settings.episodes,
                            settings.validation_seeds)
        validate_cached_fit(True, algorithm, config, settings, out)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for algorithm, seed in pool.map(fit, jobs):
            print(f'finished masked {algorithm} seed {seed}', flush=True)

    # Selected checkpoints reuse these scenarios for selection: biased, reported separately.
    rows = []
    for algorithm in args.algorithms:
        for seed in args.training_seeds:
            run = args.out/algorithm/f'seed-{seed}'
            for label, file in [('final', 'policy.pt'), ('selected', 'best_policy.pt')]:
                policy = load_masked_policy(run/file)
                rows += [run_episode(config, policy, s, name=f'{label} masked {algorithm}',
                                     training_seed=seed) for s in args.validation_seeds]
    baselines = {'random': lambda s: make_policy('random', s), 'masked random': MaskedRandomPolicy,
                 'sweep': lambda s: make_policy('sweep'), 'greedy': lambda s: make_policy('greedy')}
    for name, make in baselines.items():
        rows += [run_episode(config, make(s), s, name=name) for s in args.validation_seeds]
    write_rows(args.out/'comparison.csv', rows)
    plot_results(args.out/'comparison.csv', args.out/'comparison.png')
    for algorithm in args.algorithms:
        plot_learning([args.out/algorithm/f'seed-{s}'/'training.csv' for s in args.training_seeds],
                      args.out/f'learning-{algorithm}.png')

    print(f'\nValidation scenarios {args.validation_seeds}; RL: mean over scenarios per fit, '
          'then mean ± SD over fits. Baselines: mean ± SD over scenarios.')
    print(f'{"policy":28s} {"return":>16s} {"freshness":>10s} {"losses":>7s} {"charges":>8s}')
    for name in dict.fromkeys(r['policy'] for r in rows):
        selected = [r for r in rows if r['policy'] == name]
        seeds = list(dict.fromkeys(r['training_seed'] for r in selected))
        groups = ([[r] for r in selected] if seeds == [None] else
                  [[r for r in selected if r['training_seed'] == s] for s in seeds])

        def unit(metric):
            return [np.mean([r[metric] for r in group]) for group in groups]
        returns = unit('return')
        print(f'{name:28s} {np.mean(returns):+8.3f} ± {np.std(returns, ddof=1):5.3f} '
              f'{np.mean(unit("weighted_freshness")):10.3f} {np.mean(unit("total_losses")):7.2f} '
              f'{np.mean(unit("charge_completions")):8.2f}')


if __name__ == '__main__':
    main()
