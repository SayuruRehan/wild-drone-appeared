"""Teacher-assisted collection diagnostic; deployment is an unshielded learner."""

import copy
import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from wild_drone.baselines import GreedyPolicy
from wild_drone.behavior import TOLERANCE, return_status
from wild_drone.config import EnvConfig
from wild_drone.coverage_probe import merge_data, model_digest, save_dataset, validate_ids
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
from wild_drone.training_io import atomic_csv_save, atomic_json_save

FORMAT = 'wild-drone-recovery-coverage-v1'
METHOD = 'supervised-recovery-coverage-probe'
ARMS = ('learner', 'assisted')
COLLECTION = {'teacher_probability': .5, 'return_margin_move_costs': 6.}


def collect_recovery_rollouts(config, scenarios, policy, arm, collection_seed, *,
                              teacher_probability=.5, return_margin_move_costs=6.):
    validate_ids(scenarios)
    validate_ids([collection_seed])
    if (arm not in ARMS or not 0 <= teacher_probability <= 1 or
            not 0 <= return_margin_move_costs < float('inf')):
        raise ValueError('invalid collection arm, teacher probability or return margin')
    teacher, env = GreedyPolicy(), WildfireFleetEnv(config)
    rng = np.random.default_rng(collection_seed)  # Never use the environment's live RNG.
    observations, records, audit = [], [], []
    for scenario in scenarios:
        observation, _ = env.reset(seed=scenario)
        done = False
        while not done:
            status = return_status(env.positions, env.batteries, env.lost, env.base, config)
            charge = (~env.lost) & (status['distances'] == 0) & \
                (env.batteries < config.battery_capacity)
            risk = (~env.lost) & (status['distances'] > 0) & \
                (status['margin'] <= return_margin_move_costs*config.move_cost+TOLERANCE)
            label = teacher.act(env, observation)
            learner_action = policy.act(env, observation)
            coin = rng.random() < teacher_probability
            teacher_execution = arm == 'assisted' and (risk.any() or charge.any() or coin)
            executed = label if teacher_execution else learner_action
            observations.append(observation)
            records.append({'actions': label, 'scenario_seeds': scenario, 'steps': env.elapsed,
                            'critical': status['critical'].copy(), 'live': (~env.lost).copy(),
                            'needs_charge': charge.copy(), 'positions': env.positions.copy(),
                            'base': env.base.copy()})
            row = {'scenario_seed': scenario, 'step': env.elapsed, 'teacher_action': label,
                   'learner_action': learner_action, 'executed_action': executed,
                   'teacher_execution': bool(teacher_execution), 'risk_gate': bool(risk.any()),
                   'charging_gate': bool(charge.any()), 'random_teacher_choice': bool(coin),
                   'live_drones': int((~env.lost).sum()),
                   'critical_drones': int(status['critical'].sum()),
                   'unreachable_drones': int(status['unreachable'].sum()),
                   'charging_drones': int(charge.sum()), 'all_stay_label': label == 0}
            observation, _, terminal, truncated, info = env.step(executed)
            row |= {'new_safe_returns': info['safe_returns'],
                    'new_charge_completions': info['charge_completions'],
                    'new_losses': info['new_losses']}
            audit.append(row)
            done = terminal or truncated
    maps, vector = tensor_observations(observations)
    return ({'env_config': config.to_dict(), 'maps': maps, 'vector': vector,
             **{k: torch.from_numpy(np.stack([r[k] for r in records])) for k in records[0]}}, audit)


def load_recovery(path, record=None):
    path = Path(path)
    if record and hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('recovery checkpoint bytes differ from verified manifest')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved.get('format') != FORMAT or saved.get('method') != METHOD or
            saved.get('arm') not in ARMS):
        raise ValueError('requires explicitly labeled supervised recovery checkpoint')
    if record and (saved['completed_round'] != record['completed_round'] or
                   saved['completed_updates'] != record['completed_updates'] or
                   model_digest(saved['state_dict']) != record['model_sha256']):
        raise ValueError('recovery checkpoint identity differs from verified manifest')
    config = EnvConfig(**saved['env_config'])
    model = FleetNetwork(config.grid_size)
    model.load_state_dict(saved['state_dict'])
    return ProbePolicy(model, config)


def fit_recovery(config, initial, validation, scenario_batches, settings, seed, arm, output_dir):
    for data in (initial, validation):
        _validate_data(data, config)
    validate_ids([seed])
    if arm not in ARMS or len(scenario_batches) != settings.rounds-1:
        raise ValueError('invalid recovery arm or scenario batch count')
    added_ids = [s for batch in scenario_batches for s in batch]
    if scenario_batches:
        for batch in scenario_batches:
            validate_ids(batch)
        validate_ids(added_ids)
    train_ids, val_ids = [d['scenario_seeds'].unique().tolist() for d in (initial, validation)]
    if (set(train_ids) | set(added_ids)) & set(val_ids):
        raise ValueError('training/collection/validation scenario IDs overlap')
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError('existing recovery outputs: use a new directory; no resume')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = FleetNetwork(config.grid_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    sampler = torch.Generator().manual_seed(seed)
    metadata = write_metadata(out, config, method=METHOD, arm=arm, seed=seed,
                              coverage_config=asdict(settings), collection_config=COLLECTION,
                              device='cpu', initial_data_sha256=data_digest(initial),
                              validation_data_sha256=data_digest(validation),
                              initial_scenarios=train_ids, added_scenario_batches=scenario_batches,
                              validation_scenarios=val_ids)
    data, rows, collections, manifest, best, best_accuracy = initial, [], [], {}, None, -1.

    def export(saved, name):
        return _export(saved, out/name, progress_key='completed_updates') | {
            'completed_round': saved['completed_round'],
            'model_sha256': model_digest(saved['state_dict'])}

    for round_number in range(settings.rounds+1):
        if round_number > 1:
            batch = scenario_batches[round_number-2]
            preceding = manifest[f'round-{round_number-1}.pt']
            policy = ProbePolicy(copy.deepcopy(model), config)
            collection_seed = 100000+100*seed+round_number
            added, audit = collect_recovery_rollouts(config, batch, policy, arm, collection_seed,
                                                     **COLLECTION)
            record = save_dataset(added, out/f'added-round-{round_number}.pt') | {
                'round': round_number, 'scenario_seeds': batch,
                'collection_seed': collection_seed, 'collection_config': COLLECTION,
                'collection_policy_sha256': preceding['sha256'],
                'collection_policy_model_sha256': preceding['model_sha256'],
                'teacher_executed_steps': sum(r['teacher_execution'] for r in audit),
                'unreachable_drone_states': sum(r['unreachable_drones'] for r in audit),
                'live_drone_states': sum(r['live_drones'] for r in audit),
                'safe_returns': sum(r['new_safe_returns'] for r in audit),
                'charge_completions': sum(r['new_charge_completions'] for r in audit),
                'losses': sum(r['new_losses'] for r in audit)}
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
                    raise FloatingPointError('nonfinite recovery loss')
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm,
                                         error_if_nonfinite=True)
                optimizer.step()
        measured = {}
        for split, current in [('train', data), ('validation', validation)]:
            measured[split] = label_metrics(model, current, settings.batch_size)
            rows.append({'round': round_number,
                         'updates': round_number*settings.updates_per_round, 'split': split,
                         'states': len(current['actions']), **measured[split]})
        saved = {'format': FORMAT, 'method': METHOD, 'arm': arm, 'env_config': config.to_dict(),
                 'coverage_config': asdict(settings), 'collection_config': COLLECTION,
                 'seed': seed, 'source': metadata, 'completed_round': round_number,
                 'completed_updates': round_number*settings.updates_per_round,
                 'training_data_semantic_sha256': data_digest(data),
                 'collections': copy.deepcopy(collections),
                 'state_dict': copy.deepcopy(model.state_dict())}
        manifest[f'round-{round_number}.pt'] = export(saved, f'round-{round_number}.pt')
        if round_number and measured['validation']['joint_accuracy'] > best_accuracy:
            best_accuracy = measured['validation']['joint_accuracy']
            best = copy.deepcopy(saved)
            best['selection'] = {'metric': 'validation joint_accuracy', 'value': best_accuracy}
        atomic_csv_save(rows, out/'label_metrics.csv')
        atomic_json_save(manifest, out/'checkpoint_manifest.json')
        print(f'{arm} seed {seed}, round {round_number}/{settings.rounds}, '
              f'states {len(data["actions"])}, train {measured["train"]["joint_accuracy"]:.3f}, '
              f'validation {measured["validation"]["joint_accuracy"]:.3f}', flush=True)
    manifest['final_recovery.pt'] = export(saved, 'final_recovery.pt')
    manifest['selected_recovery.pt'] = export(best, 'selected_recovery.pt')
    atomic_json_save(manifest, out/'checkpoint_manifest.json')
    summary = {'method': METHOD, 'arm': arm, 'seed': seed, 'coverage_config': asdict(settings),
               'collection_config': COLLECTION, 'completed_updates': saved['completed_updates'],
               'final_training_states': len(data['actions']),
               'final_training_data_sha256': data_digest(data),
               'selected_round': best['completed_round'],
               'selected_validation_joint_accuracy': best_accuracy,
               'final_train_metrics': measured['train'],
               'final_validation_metrics': measured['validation']}
    atomic_json_save(summary, out/'recovery_summary.json')
    return summary
