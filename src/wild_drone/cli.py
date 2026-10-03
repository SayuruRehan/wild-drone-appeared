"""Training, paired evaluation and visual demonstration entry points."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from wild_drone.baselines import make_policy
from wild_drone.config import EnvConfig
from wild_drone.experiments import (
    plot_learning,
    plot_results,
    run_episode,
    validate_seed_split,
    write_metadata,
    write_rows,
)


def parser():
    root = argparse.ArgumentParser(description='Fully observable wildfire fleet MDP')
    sub = root.add_subparsers(dest='command', required=True)
    for name in ['demo', 'evaluate', 'train', 'compare']:
        cmd = sub.add_parser(name)
        cmd.add_argument('--config', type=Path)
        cmd.add_argument('--grid', type=int)
        cmd.add_argument('--horizon', type=int)
        cmd.add_argument('--out', type=Path, default=Path('runs')/name)
        if name in ('demo', 'evaluate', 'compare'):
            cmd.add_argument('--scenario-seeds', nargs='+', type=int,
                             default=[10000] if name == 'demo' else [10000, 10001, 10002])
            cmd.add_argument('--policy-seed', type=int, default=0)
            cmd.add_argument('--planner-horizon', type=int, default=4)
            cmd.add_argument('--planner-candidates', type=int, default=8)
            cmd.add_argument('--planner-samples', type=int, default=2)
        if name in ('demo', 'evaluate'):
            cmd.add_argument('--policy', choices=['random', 'sweep', 'greedy', 'planner'],
                             default='greedy')
            cmd.add_argument('--checkpoint', type=Path)
            cmd.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
        if name in ('train', 'compare'):
            cmd.add_argument('--episodes', type=int)
            cmd.add_argument('--warmup-steps', type=int)
            cmd.add_argument('--batch-size', type=int)
            cmd.add_argument('--train-scenario-start', type=int)
            cmd.add_argument('--device', choices=['cpu', 'cuda', 'auto'])
            cmd.add_argument('--checkpoint-interval', type=int)
            cmd.add_argument('--progress-interval', type=int)
            cmd.add_argument('--validation-interval', type=int)
            cmd.add_argument('--validation-seeds', nargs='+', type=int)
            cmd.add_argument('--spatial-encoding', action=argparse.BooleanOptionalAction)
            cmd.add_argument('--monte-carlo-baseline', action=argparse.BooleanOptionalAction)
        if name == 'train':
            cmd.add_argument('--algorithm', choices=['dqn', 'reinforce', 'a2c'], required=True)
            cmd.add_argument('--seed', type=int, default=0)
            cmd.add_argument('--resume', type=Path)
        if name == 'compare':
            cmd.add_argument('--training-seeds', nargs='+', type=int, default=[0, 1, 2])
            cmd.add_argument('--resume-auto', action='store_true')
            cmd.add_argument('--checkpoint-selection', choices=['final', 'best'], default='final')
    plot = sub.add_parser('plot')
    plot.add_argument('--csv', type=Path, required=True)
    plot.add_argument('--out', type=Path, default=Path('runs/comparison.png'))
    curve = sub.add_parser('plot-learning')
    curve.add_argument('--csv', type=Path, nargs='+', required=True)
    curve.add_argument('--out', type=Path, default=Path('runs/learning.png'))
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'plot':
        plot_results(args.csv, args.out)
        print(f'Plot: {args.out}')
        return
    if args.command == 'plot-learning':
        plot_learning(args.csv, args.out)
        print(f'Learning curves: {args.out}')
        return
    raw = json.loads(args.config.read_text()) if args.config else {}
    config = EnvConfig(**raw.get('environment', {}))
    for option, field in [('grid', 'grid_size'), ('horizon', 'horizon')]:
        if getattr(args, option) is not None:
            config = replace(config, **{field: getattr(args, option)})
    planner = dict(horizon=getattr(args, 'planner_horizon', 4),
                   candidates=getattr(args, 'planner_candidates', 8),
                   samples=getattr(args, 'planner_samples', 2))
    if args.command in ('demo', 'evaluate'):
        if args.checkpoint:
            from wild_drone.learning import load_policy
            policy = load_policy(args.checkpoint, device=args.device)
            if not args.config and args.grid is None and args.horizon is None:
                config = policy.env_config
            if config != policy.env_config:
                raise ValueError('evaluation config must match the checkpoint environment')
            validate_seed_split(policy.train_config.scenario_seed_start,
                                policy.train_config.episodes, args.scenario_seeds)
            name, seed = policy.algorithm, policy.train_config.seed
        else:
            name, seed = args.policy, None
        rows = []
        for i, scenario in enumerate(args.scenario_seeds):
            if not args.checkpoint:
                policy = make_policy(name, args.policy_seed+scenario, **planner)
            gif = args.out/'demo.gif' if args.command == 'demo' and i == 0 else None
            rows.append(run_episode(config, policy, scenario, name=name,
                                    training_seed=seed, gif_path=gif))
        write_rows(args.out/'evaluation.csv', rows)
        write_metadata(args.out, config, command=args.command, policy=name,
                       checkpoint=str(args.checkpoint) if args.checkpoint else None,
                       scenario_seeds=args.scenario_seeds, policy_seed=args.policy_seed,
                       planner=planner)
        print(f'{args.command}: {len(rows)} episodes -> {args.out}')
        return
    from wild_drone.learning import TrainConfig, resolve_device, train
    settings = TrainConfig(**raw.get('training', {}))
    for option, field in [('episodes', 'episodes'), ('warmup_steps', 'warmup_steps'),
                          ('batch_size', 'batch_size'),
                          ('train_scenario_start', 'scenario_seed_start'),
                          ('device', 'device'), ('checkpoint_interval', 'checkpoint_interval'),
                          ('progress_interval', 'progress_interval'),
                          ('validation_interval', 'validation_interval'),
                          ('validation_seeds', 'validation_seeds'),
                          ('spatial_encoding', 'spatial_encoding'),
                          ('monte_carlo_baseline', 'monte_carlo_baseline')]:
        if getattr(args, option) is not None:
            settings = replace(settings, **{field: getattr(args, option)})
    if args.command == 'train':
        settings = replace(settings, seed=args.seed)
        result = train(args.algorithm, config, settings, args.out, resume_from=args.resume)
        print(f'{args.algorithm}: {result["steps"]} steps, {result["updates"]} updates -> '
              f'{args.out}')
        return
    validate_seed_split(settings.scenario_seed_start, settings.episodes, args.scenario_seeds)
    validate_seed_split(settings.scenario_seed_start, settings.episodes, settings.validation_seeds)
    if set(settings.validation_seeds) & set(args.scenario_seeds):
        raise ValueError('comparison test seeds overlap validation checkpoint-selection seeds')
    if args.checkpoint_selection == 'best' and not settings.validation_seeds:
        raise ValueError('best checkpoint selection requires validation seeds')
    if len(set(args.training_seeds)) != len(args.training_seeds):
        raise ValueError('training seeds must be distinct')
    import torch

    from wild_drone.training_io import validate_resume
    resolve_device(settings.device)
    for algorithm in ('dqn', 'reinforce', 'a2c'):
        for seed in args.training_seeds:
            current = replace(settings, seed=seed)
            out = args.out/algorithm/f'seed-{seed}'
            state = out/'training_state.pt'
            if state.exists():
                if not args.resume_auto:
                    raise ValueError('existing training state: use --resume-auto or another output')
                saved = torch.load(state, map_location='cpu', weights_only=True)
                validate_resume(saved, algorithm, config, current)
            elif (out/'policy.pt').exists():
                raise ValueError('existing policy has no resume state: use another output')
    rows = []
    for name in ('random', 'sweep', 'greedy', 'planner'):
        for scenario in args.scenario_seeds:
            policy = make_policy(name, args.policy_seed+scenario, **planner)
            rows.append(run_episode(config, policy, scenario, name=name))
    for algorithm in ('dqn', 'reinforce', 'a2c'):
        for seed in args.training_seeds:
            out = args.out/algorithm/f'seed-{seed}'
            current = replace(settings, seed=seed)
            state = out/'training_state.pt'
            result = train(algorithm, config, current, out,
                           resume_from=state if args.resume_auto and state.exists() else None)
            from wild_drone.learning import load_policy
            file = 'best_policy.pt' if args.checkpoint_selection == 'best' else 'policy.pt'
            policy = load_policy(out/file, device=current.device)
            for scenario in args.scenario_seeds:
                rows.append(run_episode(config, policy, scenario, name=algorithm,
                                        training_seed=seed))
            print(f'{algorithm} seed {seed}: {result["updates"]} updates')
    write_rows(args.out/'comparison.csv', rows)
    write_metadata(args.out, config, command='compare', training_seeds=args.training_seeds,
                   scenario_seeds=args.scenario_seeds, train_config=settings.to_dict(),
                   planner=planner, policy_seed=args.policy_seed,
                   checkpoint_selection=args.checkpoint_selection)
    print(f'Comparison: {len(rows)} rows -> {args.out}')


if __name__ == '__main__':
    main()
