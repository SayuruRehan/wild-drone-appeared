"""Matched supervised coverage/DAgger diagnostic; never an RL training path."""

import copy
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from wild_drone.baselines import GreedyPolicy
from wild_drone.behavior import return_status
from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.experiments import write_metadata
from wild_drone.imitation_probe import (
    ProbePolicy,
    _export,
    _validate_data,
    data_digest,
    label_metrics,
)
from wild_drone.learning import FleetNetwork, tensor_observations
from wild_drone.training_io import atomic_csv_save, atomic_json_save, atomic_torch_save

FORMAT = 'wild-drone-greedy-coverage-v1'
METHOD = 'supervised-greedy-coverage-probe'
ARMS = ('fixed', 'teacher', 'dagger')


def validate_ids(ids):
    if (not ids or len(set(ids)) != len(ids) or
            any(not isinstance(s, int) or isinstance(s, bool) or s < 0 for s in ids)):
        raise ValueError('scenario/fit IDs must be distinct nonnegative integers')


@dataclass(frozen=True)
class CoverageConfig:
    rounds: int = 5
    updates_per_round: int = 1000
    batch_size: int = 64
    learning_rate: float = .001
    max_grad_norm: float = .5

    def __post_init__(self):
        for key in ('rounds', 'updates_per_round', 'batch_size'):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        for key in ('learning_rate', 'max_grad_norm'):
            if not 0 < getattr(self, key) < float('inf'):
                raise ValueError(f'{key} must be finite and positive')


def collect_rollouts(config, scenarios, policy=None):
    """Query Greedy on real visited states, while executing the supplied policy."""
    scenarios = tuple(scenarios)
    validate_ids(scenarios)
    teacher, env = GreedyPolicy(), WildfireFleetEnv(config)
    observations, records, audit = [], [], []
    for scenario in scenarios:
        observation, _ = env.reset(seed=scenario)
        done = False
        while not done:
            status = return_status(env.positions, env.batteries, env.lost, env.base, config)
            label = teacher.act(env, observation)
            executed = label if policy is None else policy.act(env, observation)
            charge = (~env.lost) & np.all(env.positions == env.base, axis=1) & \
                (env.batteries < config.battery_capacity)
            observations.append(observation)
            records.append({'actions': label, 'scenario_seeds': scenario, 'steps': env.elapsed,
                            'critical': status['critical'].copy(), 'live': (~env.lost).copy(),
                            'needs_charge': charge.copy(), 'positions': env.positions.copy(),
                            'base': env.base.copy()})
            audit.append({'scenario_seed': scenario, 'step': env.elapsed,
                          'teacher_action': label, 'executed_action': executed,
                          'live_drones': int((~env.lost).sum()),
                          'critical_drones': int(status['critical'].sum()),
                          'unreachable_drones': int(status['unreachable'].sum()),
                          'charging_drones': int(charge.sum()), 'all_stay_label': label == 0})
            observation, _, terminal, truncated, _ = env.step(executed)
            done = terminal or truncated
    maps, vector = tensor_observations(observations)
    return ({'env_config': config.to_dict(), 'maps': maps, 'vector': vector,
             **{key: torch.from_numpy(np.stack([r[key] for r in records])) for key in records[0]}},
            audit)


def merge_data(a, b):
    if a['env_config'] != b['env_config'] or a.keys() != b.keys():
        raise ValueError('cannot merge teacher data with different environment/schema')
    return {'env_config': a['env_config'],
            **{k: torch.cat((a[k], b[k])) for k in a if k != 'env_config'}}


def model_digest(state):
    return data_digest({'env_config': {}, **state})


def save_dataset(data, path):
    atomic_torch_save(data, path)
    restored = torch.load(path, map_location='cpu', weights_only=True)
    semantic = data_digest(data)
    if data_digest(restored) != semantic:
        raise ValueError('coverage dataset export differs')
    return {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'semantic_sha256': semantic, 'states': len(data['actions']),
            'critical_drone_states': int(data['critical'].sum()),
            'charging_drone_states': int(data['needs_charge'].sum()),
            'joint_action_counts': torch.bincount(data['actions'], minlength=125).tolist()}


def load_coverage(path, record=None):
    path = Path(path)
    if record and hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('coverage checkpoint bytes differ from verified manifest')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved.get('format') != FORMAT or saved.get('method') != METHOD or
            saved.get('arm') not in ARMS):
        raise ValueError('requires explicitly labeled supervised coverage checkpoint')
    if record and (saved['completed_round'] != record['completed_round'] or
                   saved['completed_updates'] != record['completed_updates'] or
                   model_digest(saved['state_dict']) != record['model_sha256']):
        raise ValueError('coverage checkpoint identity differs from verified manifest')
    config = EnvConfig(**saved['env_config'])
    model = FleetNetwork(config.grid_size)
    model.load_state_dict(saved['state_dict'])
    return ProbePolicy(model, config)


def fit_coverage(config, initial, validation, scenario_batches, settings, seed, arm, output_dir):
    for data in (initial, validation):
        _validate_data(data, config)
    validate_ids([seed])
    if arm not in ARMS or (len(scenario_batches) != settings.rounds-1 and
                           not (arm == 'fixed' and not scenario_batches)):
        raise ValueError('invalid arm or scenario batch count')
    initial_ids = initial['scenario_seeds'].unique().tolist()
    validation_ids = validation['scenario_seeds'].unique().tolist()
    added_ids = [s for batch in scenario_batches for s in batch]
    if scenario_batches:
        for batch in scenario_batches:
            validate_ids(batch)
        validate_ids(added_ids)
    if (set(initial_ids) & set(added_ids) or
            (set(initial_ids) | set(added_ids)) & set(validation_ids)):
        raise ValueError('initial/added/validation scenario IDs overlap')
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError('existing coverage outputs: use a new directory; no resume')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = FleetNetwork(config.grid_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    sampler = torch.Generator().manual_seed(seed)
    metadata = write_metadata(out, config, method=METHOD, arm=arm, seed=seed,
                              coverage_config=asdict(settings), device='cpu',
                              initial_data_sha256=data_digest(initial),
                              validation_data_sha256=data_digest(validation),
                              initial_scenarios=initial_ids,
                              added_scenario_batches=scenario_batches,
                              validation_scenarios=validation_ids)
    data, metrics, collections, manifest = initial, [], [], {}
    best, best_accuracy = None, -1.

    def checkpoint(round_number):
        return {'format': FORMAT, 'method': METHOD, 'arm': arm, 'env_config': config.to_dict(),
                'coverage_config': asdict(settings), 'seed': seed, 'source': metadata,
                'completed_round': round_number,
                'completed_updates': round_number*settings.updates_per_round,
                'training_data_semantic_sha256': data_digest(data),
                'collections': copy.deepcopy(collections),
                'state_dict': copy.deepcopy(model.state_dict())}

    def export(saved, name):
        record = _export(saved, out/name, progress_key='completed_updates')
        return record | {'completed_round': saved['completed_round'],
                         'completed_updates': saved['completed_updates'],
                         'model_sha256': model_digest(saved['state_dict'])}

    for round_number in range(settings.rounds+1):
        if round_number > 1 and arm != 'fixed':
            batch = scenario_batches[round_number-2]
            preceding = manifest[f'round-{round_number-1}.pt']
            policy = load_coverage(out/f'round-{round_number-1}.pt', preceding) \
                if arm == 'dagger' else None
            added, audit = collect_rollouts(config, batch, policy)
            record = save_dataset(added, out/f'added-round-{round_number}.pt')
            record |= {'round': round_number, 'scenario_seeds': batch,
                       'execution': 'learner' if arm == 'dagger' else 'teacher',
                       'collection_policy_sha256': preceding['sha256'] if policy else None,
                       'unreachable_drone_states': sum(r['unreachable_drones'] for r in audit),
                       'live_drone_states': sum(r['live_drones'] for r in audit),
                       'all_stay_labels': sum(r['all_stay_label'] for r in audit)}
            collections.append(record)
            atomic_csv_save(audit, out/f'collection-round-{round_number}.csv')
            atomic_json_save(collections, out/'collections.json')
            data = merge_data(data, added)
        if round_number:
            model.train()
            for _ in range(settings.updates_per_round):
                ids = torch.randint(len(data['actions']), (settings.batch_size,), generator=sampler)
                logits, _ = model(data['maps'][ids], data['vector'][ids])
                loss = nn.functional.cross_entropy(logits, data['actions'][ids])
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite coverage loss')
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm,
                                         error_if_nonfinite=True)
                optimizer.step()
        measured = {}
        for split, current in [('train', data), ('validation', validation)]:
            measured[split] = label_metrics(model, current, settings.batch_size)
            metrics.append({'round': round_number,
                            'updates': round_number*settings.updates_per_round, 'split': split,
                            'states': len(current['actions']), **measured[split]})
        saved = checkpoint(round_number)
        manifest[f'round-{round_number}.pt'] = export(saved, f'round-{round_number}.pt')
        if round_number and measured['validation']['joint_accuracy'] > best_accuracy:
            best_accuracy = measured['validation']['joint_accuracy']
            best = copy.deepcopy(saved)
            best['selection'] = {'metric': 'validation joint_accuracy', 'value': best_accuracy}
        atomic_csv_save(metrics, out/'label_metrics.csv')
        atomic_json_save(manifest, out/'checkpoint_manifest.json')
        print(f'{arm} seed {seed}, round {round_number}/{settings.rounds}, '
              f'states {len(data["actions"])}, train {measured["train"]["joint_accuracy"]:.3f}, '
              f'validation {measured["validation"]["joint_accuracy"]:.3f}', flush=True)
    manifest['final_coverage.pt'] = export(saved, 'final_coverage.pt')
    manifest['selected_coverage.pt'] = export(best, 'selected_coverage.pt')
    atomic_json_save(manifest, out/'checkpoint_manifest.json')
    summary = {'method': METHOD, 'arm': arm, 'seed': seed, 'coverage_config': asdict(settings),
               'completed_updates': settings.rounds*settings.updates_per_round,
               'final_training_states': len(data['actions']),
               'final_training_data_sha256': data_digest(data),
               'selected_round': best['completed_round'],
               'selected_validation_joint_accuracy': best_accuracy,
               'final_train_metrics': measured['train'],
               'final_validation_metrics': measured['validation']}
    atomic_json_save(summary, out/'coverage_summary.json')
    return summary
