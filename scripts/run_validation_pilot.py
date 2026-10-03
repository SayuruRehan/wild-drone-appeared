"""Resumable repeated-seed validation study; never consumes final test scenarios."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from wild_drone.baselines import make_policy
from wild_drone.config import EnvConfig
from wild_drone.experiments import plot_learning, run_episode, validate_seed_split, write_metadata
from wild_drone.learning import TrainConfig, load_policy, resolve_device, train
from wild_drone.training_io import atomic_csv_save, atomic_json_save, validate_resume


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/development.json'))
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--training-seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--validation-seeds', nargs='+', type=int, default=list(range(5000, 5005)))
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    parser.add_argument('--algorithms', nargs='+', choices=['dqn', 'reinforce', 'a2c'],
                        default=['dqn', 'reinforce', 'a2c'])
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--double-dqn', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--out', type=Path, default=Path('runs/validation20'))
    args = parser.parse_args(argv)
    if len(set(args.algorithms)) != len(args.algorithms):
        raise ValueError('algorithms must be distinct')
    if len(set(args.training_seeds)) != len(args.training_seeds) or any(
            s < 0 for s in args.training_seeds):
        raise ValueError('training seeds must be distinct and nonnegative')
    raw = json.loads(args.config.read_text())
    config = EnvConfig(**raw['environment'])
    settings = TrainConfig(**(raw.get('training', {}) | {
        'episodes': args.episodes, 'validation_seeds': args.validation_seeds,
        'validation_interval': 20, 'checkpoint_interval': 20,
        'progress_interval': 20, 'device': args.device}))
    overrides = {key: getattr(args, key) for key in ('learning_rate', 'double_dqn')
                 if getattr(args, key) is not None}
    settings = replace(settings, **overrides)
    validate_seed_split(settings.scenario_seed_start, settings.episodes, settings.validation_seeds)
    resolve_device(settings.device)
    protocol_path = args.out/'protocol.json'
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        previous_algorithms = previous.get('algorithms', ['dqn', 'reinforce', 'a2c'])
        if (previous_algorithms != args.algorithms or
                previous['training_seeds'] != args.training_seeds):
            raise ValueError('study protocol algorithms/training seeds differ; use another output')
    # Preflight every existing run before replacing the study protocol or comparison CSV.
    for algorithm in args.algorithms:
        for seed in args.training_seeds:
            state_path = args.out/algorithm/f'seed-{seed}'/'training_state.pt'
            if state_path.exists():
                state = torch.load(state_path, map_location='cpu', weights_only=True)
                validate_resume(state, algorithm, config, replace(settings, seed=seed))
            elif (state_path.parent/'policy.pt').exists():
                raise ValueError('existing policy has no resume state: use another output')
    protocol = {'purpose': 'validation only; not final held-out evidence',
                'environment': config.to_dict(), 'training': settings.to_dict(),
                'training_seeds': args.training_seeds,
                'algorithms': args.algorithms,
                'planner': {'horizon': 3, 'candidates': 8, 'samples': 2},
                'selection': 'max scenario-mean validation return; also report final policy'}
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json_save(protocol, args.out/'protocol.json')
    write_metadata(args.out, config, **protocol)
    rows = []
    for name in ['random', 'sweep', 'greedy', 'planner']:
        rows.extend(run_episode(config, make_policy(name, seed, **protocol['planner']),
                                seed, name=name) for seed in settings.validation_seeds)
    atomic_csv_save(rows, args.out/'validation_comparison.csv')
    for algorithm in args.algorithms:
        for seed in args.training_seeds:
            out = args.out/algorithm/f'seed-{seed}'
            state = out/'training_state.pt'
            print(f'Starting/resuming {algorithm} training seed {seed}', flush=True)
            train(algorithm, config, replace(settings, seed=seed), out,
                  resume_from=state if state.exists() else None)
            for selection, file in [('final', 'policy.pt'), ('best', 'best_policy.pt')]:
                policy = load_policy(out/file, device=args.device)
                rows.extend(run_episode(config, policy, scenario,
                                        name=f'{selection}-{algorithm}', training_seed=seed)
                            for scenario in settings.validation_seeds)
            atomic_csv_save(rows, args.out/'validation_comparison.csv')
        plot_learning([args.out/algorithm/f'seed-{seed}'/'training.csv'
                       for seed in args.training_seeds], args.out/f'{algorithm}-learning.png')
    print(f'Validation pilot complete: {len(rows)} rows -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
