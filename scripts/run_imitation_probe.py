"""Bounded supervised capability diagnostic; no RL training or simulator changes."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from wild_drone.baselines import GreedyPolicy
from wild_drone.behavior import behavior_episode
from wild_drone.config import EnvConfig
from wild_drone.imitation_probe import (
    METHOD,
    ProbeConfig,
    collect_teacher_data,
    data_digest,
    fit_probe,
    load_probe,
)
from wild_drone.training_io import atomic_csv_save, atomic_json_save, atomic_torch_save


def run_probe(config, training_scenarios, validation_scenarios, fit_seeds, settings, out):
    out = Path(out)
    for ids in (training_scenarios, validation_scenarios, fit_seeds):
        if (not ids or len(set(ids)) != len(ids) or
                any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
                    for seed in ids)):
            raise ValueError('scenario/fit seeds must be distinct nonnegative integers')
    if set(training_scenarios) & set(validation_scenarios):
        raise ValueError('training/validation scenario IDs overlap')
    if out.exists() and any(out.iterdir()):
        raise ValueError('existing probe outputs: use a new directory')
    torch.set_num_threads(1)
    training = collect_teacher_data(config, training_scenarios)
    validation = collect_teacher_data(config, validation_scenarios)
    out.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for split, data in [('training', training), ('validation', validation)]:
        path = out/f'{split}_teacher_data.pt'
        atomic_torch_save(data, path)
        restored = torch.load(path, map_location='cpu', weights_only=True)
        if data_digest(restored) != data_digest(data):
            raise ValueError(f'{split} teacher-data export differs')
        counts = torch.bincount(data['actions'], minlength=125)
        datasets[split] = {'file': path.name,
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'semantic_sha256': data_digest(data), 'states': len(data['actions']),
                           'critical_drone_states': int(data['critical'].sum()),
                           'charging_drone_states': int(data['needs_charge'].sum()),
                           'joint_action_counts': counts.tolist()}
    source_root = Path(__file__).resolve().parents[1]
    source_paths = ['scripts/run_imitation_probe.py', 'src/wild_drone/imitation_probe.py',
                    'src/wild_drone/learning.py', 'src/wild_drone/environment.py',
                    'src/wild_drone/baselines.py', 'src/wild_drone/behavior.py',
                    'src/wild_drone/config.py', 'src/wild_drone/training_io.py',
                    'src/wild_drone/experiments.py', 'docs/imitation-probe-plan.md']
    protocol = {'method': METHOD, 'environment': config.to_dict(),
                'training_scenarios': list(training_scenarios),
                'validation_scenarios': list(validation_scenarios), 'fit_seeds': list(fit_seeds),
                'probe_config_except_seed': {k: v for k, v in settings.to_dict().items()
                                             if k != 'seed'},
                'datasets': datasets, 'device': 'cpu',
                'selection': 'maximum validation joint agreement, earliest measured epoch tie; '
                             'initial checkpoint ineligible',
                'teacher': 'GreedyPolicy on real full-observation simulator trajectories',
                'interpretation': 'Supervised logits, not Q values; teacher-state label agreement '
                                  'and unshielded autonomous mission performance are separate.',
                'source_sha256': {p: hashlib.sha256((source_root/p).read_bytes()).hexdigest()
                                  for p in source_paths}}
    atomic_json_save(protocol, out/'protocol.json')
    rows, fit_summaries = [], []
    for seed in fit_seeds:
        started = time.perf_counter()
        current = ProbeConfig(**(settings.to_dict() | {'seed': seed}))
        directory = out/f'seed-{seed}'
        summary = fit_probe(config, training, validation, current, directory)
        fit_summaries.append(summary | {'fit_seconds': time.perf_counter()-started})
        manifest = json.loads((directory/'probe_manifest.json').read_text())
        for checkpoint in ['initial_probe.pt', 'final_probe.pt', 'selected_probe.pt']:
            path = directory/checkpoint
            record = manifest[checkpoint]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                raise ValueError(f'{path}: checkpoint bytes differ from verified manifest')
            policy = load_probe(path)
            for scenario in validation_scenarios:
                keep_trace = seed == fit_seeds[0] and checkpoint == 'final_probe.pt' and \
                    scenario == validation_scenarios[0]
                row, trace = behavior_episode(config, policy, scenario, name=METHOD,
                                               training_seed=seed, trace=keep_trace)
                common = {'checkpoint': checkpoint, 'completed_epochs': record['completed_epochs'],
                          'checkpoint_sha256': record['sha256']}
                rows.append(row | common)
                if keep_trace:
                    atomic_csv_save([r | common | {'training_seed': seed, 'scenario_seed': scenario}
                                     for r in trace], out/'example-trace.csv')
            atomic_csv_save(rows, out/'validation_missions.csv')
        atomic_json_save(fit_summaries, out/'fits.json')
    for scenario in validation_scenarios:
        row, _ = behavior_episode(config, GreedyPolicy(), scenario, name='greedy')
        rows.append({key: row.get(key) for key in rows[0]})
    atomic_csv_save(rows, out/'validation_missions.csv')
    print(f'Supervised probe complete: {len(rows)} autonomous validation missions -> {out}',
          flush=True)
    return protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/development.json'))
    parser.add_argument('--training-scenarios', nargs='+', type=int, default=list(range(20)))
    parser.add_argument('--validation-scenarios', nargs='+', type=int,
                        default=list(range(5000, 5005)))
    parser.add_argument('--fit-seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    config = EnvConfig(**json.loads(args.config.read_text())['environment'])
    settings = ProbeConfig(epochs=args.epochs, batch_size=args.batch_size,
                           learning_rate=args.learning_rate)
    run_probe(config, args.training_scenarios, args.validation_scenarios, args.fit_seeds,
              settings, args.out)


if __name__ == '__main__':
    main()
