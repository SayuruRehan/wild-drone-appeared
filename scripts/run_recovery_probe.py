"""Compare learner-only and teacher-assisted real rollout collection."""

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from wild_drone.baselines import GreedyPolicy
from wild_drone.behavior import behavior_episode
from wild_drone.config import EnvConfig
from wild_drone.coverage_probe import CoverageConfig, collect_rollouts, save_dataset, validate_ids
from wild_drone.recovery_probe import ARMS, COLLECTION, METHOD, fit_recovery, load_recovery
from wild_drone.training_io import atomic_csv_save, atomic_json_save


def run_study(config, initial_scenarios, scenario_batches, validation_scenarios, fit_seeds,
              settings, out):
    out = Path(out)
    for ids in (initial_scenarios, validation_scenarios, fit_seeds):
        validate_ids(ids)
    if len(scenario_batches) != settings.rounds-1:
        raise ValueError('scenario batch count differs from rounds')
    for batch in scenario_batches:
        validate_ids(batch)
    added_ids = [s for batch in scenario_batches for s in batch]
    if added_ids:
        validate_ids(added_ids)
    if set(added_ids) != set(initial_scenarios):
        raise ValueError('collection must revisit the same initial training worlds exactly once')
    if set(initial_scenarios) & set(validation_scenarios):
        raise ValueError('training/validation scenario IDs overlap')
    if out.exists() and any(out.iterdir()):
        raise ValueError('existing recovery outputs: use a new directory; no resume')
    torch.set_num_threads(1)
    initial, initial_audit = collect_rollouts(config, initial_scenarios)
    validation, validation_audit = collect_rollouts(config, validation_scenarios)
    out.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[1]
    sources = ['scripts/run_recovery_probe.py', 'src/wild_drone/recovery_probe.py',
               'src/wild_drone/coverage_probe.py', 'src/wild_drone/imitation_probe.py',
               'src/wild_drone/learning.py', 'src/wild_drone/environment.py',
               'src/wild_drone/baselines.py', 'src/wild_drone/behavior.py',
               'src/wild_drone/config.py', 'src/wild_drone/experiments.py',
               'src/wild_drone/training_io.py', 'docs/recovery-protocol.md']
    expected = (len(ARMS)*len(fit_seeds)*(settings.rounds+1)+1)*len(validation_scenarios)
    protocol = {'study': 'teacher-assisted-recovery-collection', 'method': METHOD,
                'environment': config.to_dict(), 'coverage_config': asdict(settings),
                'collection_config': COLLECTION, 'arms': list(ARMS),
                'initial_scenarios': list(initial_scenarios), 'scenario_batches': scenario_batches,
                'validation_scenarios': list(validation_scenarios), 'fit_seeds': list(fit_seeds),
                'expected_missions': expected, 'device': 'cpu',
                'initial_data': save_dataset(initial, out/'training.pt'),
                'validation_data': save_dataset(validation, out/'validation.pt'),
                'source_sha256': {p: hashlib.sha256((source_root/p).read_bytes()).hexdigest()
                                  for p in sources}}
    atomic_json_save(protocol, out/'protocol.json')
    atomic_csv_save(initial_audit, out/'training-collection.csv')
    atomic_csv_save(validation_audit, out/'validation-collection.csv')
    rows, summaries, paired_initial, paired_first = [], [], {}, {}
    for arm in ARMS:
        for seed in fit_seeds:
            directory = out/arm/f'seed-{seed}'
            started = time.perf_counter()
            summary = fit_recovery(config, initial, validation, scenario_batches, settings,
                                   seed, arm, directory)
            summaries.append(summary | {'fit_seconds': time.perf_counter()-started})
            manifest = json.loads((directory/'checkpoint_manifest.json').read_text())
            for round_number, pairs in [(0, paired_initial), (1, paired_first)]:
                digest = manifest[f'round-{round_number}.pt']['model_sha256']
                if seed in pairs and digest != pairs[seed]:
                    raise ValueError('initial/shared first-round model tensors differ across arms')
                pairs[seed] = digest
            paths = [f'round-{r}.pt' for r in range(1, settings.rounds+1)]
            paths += ['selected_recovery.pt']
            for checkpoint in paths:
                record = manifest[checkpoint]
                policy = load_recovery(directory/checkpoint, record)
                for scenario in validation_scenarios:
                    keep = (arm == 'assisted' and seed == fit_seeds[0] and
                            checkpoint == f'round-{settings.rounds}.pt' and
                            scenario == validation_scenarios[0])
                    row, trace = behavior_episode(config, policy, scenario, name=METHOD,
                                                   training_seed=seed, trace=keep)
                    common = {'arm': arm, 'checkpoint': checkpoint,
                              'completed_round': record['completed_round'],
                              'completed_updates': record['completed_updates'],
                              'checkpoint_sha256': record['sha256']}
                    rows.append(row | common)
                    if keep:
                        atomic_csv_save([r | common | {'training_seed': seed,
                                                       'scenario_seed': scenario} for r in trace],
                                        out/'example-trace.csv')
                atomic_csv_save(rows, out/'validation_missions.csv')
            atomic_json_save(summaries, out/'fits.json')
    for scenario in validation_scenarios:
        row, _ = behavior_episode(config, GreedyPolicy(), scenario, name='greedy')
        rows.append({key: row.get(key) for key in rows[0]})
    atomic_csv_save(rows, out/'validation_missions.csv')
    persisted = list(csv.DictReader((out/'validation_missions.csv').open()))
    if len(persisted) != expected or sum(r['policy'] == 'greedy' for r in persisted) != len(
            validation_scenarios):
        raise ValueError('persisted mission count or baseline rows differ from protocol')
    atomic_json_save({'persisted_missions': len(persisted),
                      'mission_csv_sha256': hashlib.sha256(
                          (out/'validation_missions.csv').read_bytes()).hexdigest(),
                      'paired_initial_model_sha256': paired_initial,
                      'paired_first_round_model_sha256': paired_first}, out/'completion.json')
    print(f'Recovery probe complete: {len(persisted)} persisted missions -> {out}', flush=True)
    return protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/development.json'))
    parser.add_argument('--training-scenarios', nargs='+', type=int, default=list(range(100)))
    parser.add_argument('--validation-scenarios', nargs='+', type=int,
                        default=list(range(5000, 5005)))
    parser.add_argument('--fit-seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--updates-per-round', type=int, default=4000)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    settings = CoverageConfig(rounds=args.rounds, updates_per_round=args.updates_per_round)
    groups, worlds = settings.rounds-1, args.training_scenarios
    if groups < 1 or len(worlds) < groups:
        parser.error('at least two rounds and one world per collection batch are required')
    batches = [worlds[i*len(worlds)//groups:(i+1)*len(worlds)//groups] for i in range(groups)]
    config = EnvConfig(**json.loads(args.config.read_text())['environment'])
    run_study(config, worlds, batches, args.validation_scenarios, args.fit_seeds, settings,
              args.out)


if __name__ == '__main__':
    main()
