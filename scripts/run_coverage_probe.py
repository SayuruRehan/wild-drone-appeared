"""Compare broader Greedy trajectories and learner-state labels at matched update counts."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from wild_drone.baselines import GreedyPolicy
from wild_drone.behavior import behavior_episode
from wild_drone.config import EnvConfig
from wild_drone.coverage_probe import (
    ARMS,
    METHOD,
    CoverageConfig,
    collect_rollouts,
    fit_coverage,
    load_coverage,
    save_dataset,
    validate_ids,
)
from wild_drone.training_io import atomic_csv_save, atomic_json_save


def run_study(config, initial_scenarios, added_batches, validation_scenarios, fit_seeds,
              settings, out):
    out = Path(out)
    for ids in [initial_scenarios, validation_scenarios, fit_seeds, *added_batches]:
        validate_ids(ids)
    added_ids = [s for batch in added_batches for s in batch]
    if added_ids:
        validate_ids(added_ids)
    if len(added_batches) != settings.rounds-1:
        raise ValueError('added batch count must be rounds minus one')
    if (set(initial_scenarios) & set(added_ids) or
            (set(initial_scenarios) | set(added_ids)) & set(validation_scenarios)):
        raise ValueError('initial/added/validation scenario IDs overlap')
    if out.exists() and any(out.iterdir()):
        raise ValueError('existing coverage outputs: use a new directory')
    torch.set_num_threads(1)
    initial, initial_audit = collect_rollouts(config, initial_scenarios)
    validation, validation_audit = collect_rollouts(config, validation_scenarios)
    out.mkdir(parents=True, exist_ok=True)
    sources = ['scripts/run_coverage_probe.py', 'src/wild_drone/coverage_probe.py',
               'src/wild_drone/imitation_probe.py', 'src/wild_drone/learning.py',
               'src/wild_drone/environment.py', 'src/wild_drone/baselines.py',
               'src/wild_drone/behavior.py', 'src/wild_drone/config.py',
               'src/wild_drone/experiments.py', 'src/wild_drone/training_io.py',
               'docs/coverage-probe-protocol.md']
    source_root = Path(__file__).resolve().parents[1]
    protocol = {'method': METHOD, 'environment': config.to_dict(),
                'coverage_config': asdict(settings), 'initial_scenarios': list(initial_scenarios),
                'added_batches': added_batches, 'validation_scenarios': list(validation_scenarios),
                'fit_seeds': list(fit_seeds), 'arms': list(ARMS), 'device': 'cpu',
                'initial_data': save_dataset(initial, out/'initial_teacher_data.pt'),
                'validation_data': save_dataset(validation, out/'validation_teacher_data.pt'),
                'source_sha256': {p: hashlib.sha256((source_root/p).read_bytes()).hexdigest()
                                  for p in sources}}
    atomic_json_save(protocol, out/'protocol.json')
    atomic_csv_save(initial_audit, out/'initial-collection.csv')
    atomic_csv_save(validation_audit, out/'validation-collection.csv')
    rows, summaries, common_models = [], [], {}
    for arm in ARMS:
        for seed in fit_seeds:
            directory = out/arm/f'seed-{seed}'
            started = time.perf_counter()
            summary = fit_coverage(config, initial, validation, added_batches,
                                   settings, seed, arm, directory)
            summaries.append(summary | {'fit_and_collection_seconds': time.perf_counter()-started})
            manifest = json.loads((directory/'checkpoint_manifest.json').read_text())
            # Check matched initial/first-round weights without claiming more independent fits.
            for round_number in [0, 1]:
                digest = manifest[f'round-{round_number}.pt']['model_sha256']
                key = (seed, round_number)
                if key in common_models and common_models[key] != digest:
                    raise ValueError('matched arms differ before their data-coverage intervention')
                common_models[key] = digest
            paths = [f'round-{r}.pt' for r in range(1, settings.rounds+1)]
            paths += ['selected_coverage.pt']
            for checkpoint in paths:
                record = manifest[checkpoint]
                policy = load_coverage(directory/checkpoint, record)
                for scenario in validation_scenarios:
                    keep_trace = (arm == 'dagger' and seed == fit_seeds[0] and
                                  checkpoint == f'round-{settings.rounds}.pt' and
                                  scenario == validation_scenarios[0])
                    row, trace = behavior_episode(config, policy, scenario, name=METHOD,
                                                   training_seed=seed, trace=keep_trace)
                    common = {'arm': arm, 'checkpoint': checkpoint,
                              'completed_round': record['completed_round'],
                              'completed_updates': record['completed_updates'],
                              'checkpoint_sha256': record['sha256']}
                    rows.append(row | common)
                    if keep_trace:
                        atomic_csv_save([r | common | {'training_seed': seed,
                                                      'scenario_seed': scenario} for r in trace],
                                        out/'example-trace.csv')
                atomic_csv_save(rows, out/'validation_missions.csv')
            atomic_json_save(summaries, out/'fits.json')
    atomic_json_save({f'{seed}/round-{r}': h for (seed, r), h in common_models.items()},
                     out/'matched-models.json')
    for scenario in validation_scenarios:
        row, _ = behavior_episode(config, GreedyPolicy(), scenario, name='greedy')
        rows.append({key: row.get(key) for key in rows[0]})
    atomic_csv_save(rows, out/'validation_missions.csv')
    print(f'Coverage probe complete: {len(rows)} autonomous missions -> {out}', flush=True)
    return protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/development.json'))
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--updates-per-round', type=int, default=1000)
    parser.add_argument('--scenarios-per-round', type=int, default=20)
    parser.add_argument('--fit-seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--validation-scenarios', nargs='+', type=int,
                        default=list(range(5000, 5005)))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.scenarios_per_round < 1:
        raise ValueError('scenarios per round must be positive')
    config = EnvConfig(**json.loads(args.config.read_text())['environment'])
    settings = CoverageConfig(rounds=args.rounds, updates_per_round=args.updates_per_round)
    count = args.scenarios_per_round
    batches = [list(range(r*count, (r+1)*count)) for r in range(args.rounds)]
    run_study(config, batches[0], batches[1:], args.validation_scenarios,
              args.fit_seeds, settings, args.out)


if __name__ == '__main__':
    main()
