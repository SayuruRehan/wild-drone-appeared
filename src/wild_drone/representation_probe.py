"""Matched supervised derived-input experiment; not an RL training path."""

import copy
import hashlib
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from wild_drone.config import EnvConfig
from wild_drone.coverage_probe import model_digest, validate_ids
from wild_drone.experiments import write_metadata
from wild_drone.imitation_probe import (
    ProbePolicy,
    _export,
    _validate_data,
    data_digest,
    label_metrics,
)
from wild_drone.learning import FleetNetwork
from wild_drone.training_io import atomic_csv_save, atomic_json_save

FORMAT = 'wild-drone-relative-energy-imitation-v1'
METHOD = 'supervised-relative-energy-probe'
ARMS = ('padded-control', 'relative-energy')


def relative_energy_features(vector, config):
    """Derived from normalized observations only; signed energy deficits are retained.

    Per drone: row/column offset divided by grid_size-1, Manhattan distance
    divided by 2*(grid_size-1), and necessary return margin divided by capacity.
    This shortest-path bound excludes extra hovering/collision costs.
    """
    drones = vector[:, :12].reshape(-1, 3, 4)
    displacement = drones[:, :, :2]-vector[:, None, 21:23]
    distance = displacement.abs().sum(-1)
    margin = drones[:, :, 2]-distance*(config.grid_size-1)*config.move_cost/config.battery_capacity
    features = torch.cat((displacement, (distance/2).unsqueeze(-1), margin.unsqueeze(-1)), -1)
    features = features.masked_fill((drones[:, :, 3] > .5).unsqueeze(-1), 0)
    return features.flatten(1)


class RepresentationNetwork(FleetNetwork):
    def __init__(self, config, arm):
        if arm not in ARMS:
            raise ValueError('invalid representation arm')
        super().__init__(config.grid_size)
        self.config, self.arm = config, arm
        self.encoder[0] = nn.Linear(self.encoder[0].in_features+12, 128)

    def extra_features(self, vector):
        return (relative_energy_features(vector, self.config) if self.arm == 'relative-energy'
                else vector.new_zeros((len(vector), 12)))

    def forward(self, maps, vector):
        return super().forward(maps, torch.cat((vector, self.extra_features(vector)), 1))


def load_representation(path, record=None):
    path = Path(path)
    if record and hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('representation checkpoint bytes differ from verified manifest')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved.get('format') != FORMAT or saved.get('method') != METHOD or
            saved.get('arm') not in ARMS):
        raise ValueError('requires explicitly labeled supervised representation checkpoint')
    if record and (saved['completed_round'] != record['completed_round'] or
                   saved['completed_updates'] != record['completed_updates'] or
                   model_digest(saved['state_dict']) != record['model_sha256']):
        raise ValueError('representation checkpoint identity differs from verified manifest')
    config = EnvConfig(**saved['env_config'])
    model = RepresentationNetwork(config, saved['arm'])
    model.load_state_dict(saved['state_dict'])
    return ProbePolicy(model, config)


def fit_representation(config, training, validation, settings, seed, arm, output_dir):
    for data in (training, validation):
        _validate_data(data, config)
    validate_ids([seed])
    if arm not in ARMS:
        raise ValueError('invalid representation arm')
    train_ids, val_ids = [d['scenario_seeds'].unique().tolist() for d in (training, validation)]
    if set(train_ids) & set(val_ids):
        raise ValueError('training/validation scenario IDs overlap')
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError('existing representation outputs: use a new directory; no resume')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = RepresentationNetwork(config, arm)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    sampler = torch.Generator().manual_seed(seed)
    train_hash, val_hash = data_digest(training), data_digest(validation)
    metadata = write_metadata(out, config, method=METHOD, arm=arm, seed=seed,
                              coverage_config=asdict(settings), device='cpu',
                              training_data_sha256=train_hash, validation_data_sha256=val_hash,
                              training_scenarios=train_ids, validation_scenarios=val_ids)
    rows, manifest, best, best_accuracy = [], {}, None, -1.

    def export(saved, name):
        return _export(saved, out/name, progress_key='completed_updates') | {
            'completed_round': saved['completed_round'],
            'model_sha256': model_digest(saved['state_dict'])}

    for round_number in range(settings.rounds+1):
        if round_number:
            model.train()
            for _ in range(settings.updates_per_round):
                ids = torch.randint(len(training['actions']), (settings.batch_size,),
                                    generator=sampler)
                logits, _ = model(training['maps'][ids], training['vector'][ids])
                loss = nn.functional.cross_entropy(logits, training['actions'][ids])
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite representation loss')
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm,
                                         error_if_nonfinite=True)
                optimizer.step()
        measured = {}
        for split, data in [('train', training), ('validation', validation)]:
            measured[split] = label_metrics(model, data, settings.batch_size)
            rows.append({'round': round_number,
                         'updates': round_number*settings.updates_per_round, 'split': split,
                         'states': len(data['actions']), **measured[split]})
        saved = {'format': FORMAT, 'method': METHOD, 'arm': arm, 'env_config': config.to_dict(),
                 'coverage_config': asdict(settings), 'seed': seed, 'source': metadata,
                 'completed_round': round_number,
                 'completed_updates': round_number*settings.updates_per_round,
                 'training_data_semantic_sha256': train_hash,
                 'validation_data_semantic_sha256': val_hash,
                 'state_dict': copy.deepcopy(model.state_dict())}
        manifest[f'round-{round_number}.pt'] = export(saved, f'round-{round_number}.pt')
        if round_number and measured['validation']['joint_accuracy'] > best_accuracy:
            best_accuracy = measured['validation']['joint_accuracy']
            best = copy.deepcopy(saved)
            best['selection'] = {'metric': 'validation joint_accuracy', 'value': best_accuracy}
        atomic_csv_save(rows, out/'label_metrics.csv')
        atomic_json_save(manifest, out/'checkpoint_manifest.json')
        print(f'{arm} seed {seed}, round {round_number}/{settings.rounds}, '
              f'train {measured["train"]["joint_accuracy"]:.3f}, '
              f'validation {measured["validation"]["joint_accuracy"]:.3f}', flush=True)
    manifest['final_representation.pt'] = export(saved, 'final_representation.pt')
    manifest['selected_representation.pt'] = export(best, 'selected_representation.pt')
    atomic_json_save(manifest, out/'checkpoint_manifest.json')
    summary = {'method': METHOD, 'arm': arm, 'seed': seed, 'coverage_config': asdict(settings),
               'completed_updates': settings.rounds*settings.updates_per_round,
               'training_states': len(training['actions']),
               'training_data_sha256': train_hash, 'validation_data_sha256': val_hash,
               'selected_round': best['completed_round'],
               'selected_validation_joint_accuracy': best_accuracy,
               'final_train_metrics': measured['train'],
               'final_validation_metrics': measured['validation']}
    atomic_json_save(summary, out/'representation_summary.json')
    return summary
