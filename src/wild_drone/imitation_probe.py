"""Supervised capability probe on the unchanged fleet network; this is not RL."""

import copy
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from wild_drone.baselines import GreedyPolicy
from wild_drone.behavior import return_status
from wild_drone.config import EnvConfig
from wild_drone.environment import MOVES, WildfireFleetEnv
from wild_drone.experiments import write_metadata
from wild_drone.learning import FleetNetwork, tensor_observations
from wild_drone.training_io import atomic_csv_save, atomic_json_save, atomic_torch_save

FORMAT = 'wild-drone-greedy-imitation-v1'
METHOD = 'supervised-greedy-imitation-probe'


@dataclass(frozen=True)
class ProbeConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = .001
    seed: int = 0
    evaluation_interval: int = 10
    max_grad_norm: float = .5

    def __post_init__(self):
        for name in ('epochs', 'batch_size', 'evaluation_interval'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        for name in ('learning_rate', 'max_grad_norm'):
            if not 0 < getattr(self, name) < float('inf'):
                raise ValueError(f'{name} must be finite and positive')

    def to_dict(self):
        return asdict(self)


def collect_teacher_data(config, scenarios):
    scenarios = tuple(scenarios)
    if (not scenarios or len(set(scenarios)) != len(scenarios) or
            any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
                for seed in scenarios)):
        raise ValueError('teacher scenarios must be distinct nonnegative integers')
    observations, records = [], []
    teacher, env = GreedyPolicy(), WildfireFleetEnv(config)
    for scenario in scenarios:
        observation, _ = env.reset(seed=scenario)
        done = False
        while not done:
            status = return_status(env.positions, env.batteries, env.lost, env.base, config)
            action = teacher.act(env, observation)
            observations.append(observation)
            records.append({'actions': action, 'scenario_seeds': scenario, 'steps': env.elapsed,
                            'critical': status['critical'].copy(), 'live': (~env.lost).copy(),
                            'needs_charge': ((~env.lost) &
                                             np.all(env.positions == env.base, axis=1) &
                                             (env.batteries < config.battery_capacity)).copy(),
                            'positions': env.positions.copy(), 'base': env.base.copy()})
            observation, _, terminal, truncated, _ = env.step(action)
            done = terminal or truncated
    maps, vector = tensor_observations(observations)
    return {'env_config': config.to_dict(), 'maps': maps, 'vector': vector,
            **{key: torch.from_numpy(np.stack([record[key] for record in records]))
               for key in records[0]}}


def data_digest(data):
    digest = hashlib.sha256(json.dumps(data['env_config'], sort_keys=True).encode())
    for key in sorted(set(data)-{'env_config'}):
        value = data[key].cpu().contiguous()
        digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode()+b'\0')
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _validate_data(data, config):
    count = len(data['actions'])
    shapes = {'actions': (count,), 'maps': (count, 9, config.grid_size, config.grid_size),
              'vector': (count, 24), 'critical': (count, 3), 'needs_charge': (count, 3),
              'live': (count, 3), 'positions': (count, 3, 2), 'base': (count, 2),
              'scenario_seeds': (count,), 'steps': (count,)}
    if not count or data['env_config'] != config.to_dict():
        raise ValueError('teacher data environment/size mismatch')
    for key, shape in shapes.items():
        if data[key].device.type != 'cpu' or tuple(data[key].shape) != shape:
            raise ValueError(f'teacher data {key} shape/device mismatch')
    if data['actions'].dtype != torch.int64 or not ((data['actions'] >= 0) &
                                                   (data['actions'] < 125)).all():
        raise ValueError('teacher joint-action labels must be integers in [0, 124]')


@torch.no_grad()
def label_metrics(model, data, batch_size):
    model.eval()
    logits = torch.cat([model(data['maps'][i:i+batch_size], data['vector'][i:i+batch_size])[0]
                        for i in range(0, len(data['actions']), batch_size)])
    predictions, actions = logits.argmax(1), data['actions']
    predicted_commands = torch.stack([predictions//25, (predictions//5) % 5, predictions % 5], 1)
    teacher_commands = torch.stack([actions//25, (actions//5) % 5, actions % 5], 1)
    same = predicted_commands == teacher_commands
    critical, charge, live = data['critical'], data['needs_charge'], data['live']
    proposed = data['positions']+torch.from_numpy(MOVES)[predicted_commands]
    before_distance = (data['positions']-data['base'][:, None]).abs().sum(-1)
    after_distance = (proposed-data['base'][:, None]).abs().sum(-1)

    def fraction(values, mask):
        return float(values[mask].float().mean()) if mask.any() else None

    return {'cross_entropy': float(nn.functional.cross_entropy(logits, actions)),
            'joint_accuracy': float((predictions == actions).float().mean()),
            'live_command_accuracy': fraction(same, live),
            'critical_joint_accuracy': fraction(predictions == actions, critical.any(1)),
            'critical_command_accuracy': fraction(same, critical),
            'critical_toward_base_command_fraction': fraction(after_distance < before_distance,
                                                             critical),
            'charging_hold_fraction': fraction(predicted_commands == 0, charge),
            'critical_drone_states': int(critical.sum()),
            'charging_drone_states': int(charge.sum())}


class ProbePolicy:
    def __init__(self, model, config):
        self.model, self.env_config = model.eval(), config

    def act(self, env, observation):
        if env.config != self.env_config:
            raise ValueError('probe environment differs from evaluation config')
        with torch.no_grad():
            logits, _ = self.model(*tensor_observations(observation))
        return int(logits.argmax(1).item())


def load_probe(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('format') != FORMAT or data.get('method') != METHOD:
        raise ValueError('requires an explicitly labeled imitation probe checkpoint')
    config = EnvConfig(**data['env_config'])
    model = FleetNetwork(config.grid_size)
    model.load_state_dict(data['state_dict'])
    return ProbePolicy(model, config)


def _export(checkpoint, path):
    atomic_torch_save(checkpoint, path)
    raw = path.read_bytes()
    actual = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    metadata = {key: value for key, value in checkpoint.items() if key != 'state_dict'}
    if metadata != {key: value for key, value in actual.items() if key != 'state_dict'}:
        raise ValueError(f'{path.name}: exported probe metadata differs')
    expected, saved = checkpoint['state_dict'], actual['state_dict']
    if expected.keys() != saved.keys() or any(
            expected[key].dtype != saved[key].dtype or
            not torch.equal(expected[key], saved[key]) for key in expected):
        raise ValueError(f'{path.name}: exported probe tensors differ')
    return {'completed_epochs': actual['completed_epochs'],
            'sha256': hashlib.sha256(raw).hexdigest()}


def fit_probe(config, training, validation, settings, output_dir):
    for data in (training, validation):
        _validate_data(data, config)
    train_seeds, val_seeds = (sorted(data['scenario_seeds'].unique().tolist())
                             for data in (training, validation))
    if set(train_seeds) & set(val_seeds):
        raise ValueError('training/validation teacher scenario IDs overlap')
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError('existing probe outputs: use a new directory; no resume interface')
    torch.set_num_threads(1)
    torch.manual_seed(settings.seed)
    torch.use_deterministic_algorithms(True)
    model = FleetNetwork(config.grid_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    sampler = torch.Generator().manual_seed(settings.seed)
    metadata = write_metadata(out, config, method=METHOD, probe_config=settings.to_dict(),
                              training_scenarios=train_seeds, validation_scenarios=val_seeds,
                              device='cpu', training_data_sha256=data_digest(training),
                              validation_data_sha256=data_digest(validation))

    def checkpoint(epoch):
        return {'format': FORMAT, 'method': METHOD, 'env_config': config.to_dict(),
                'probe_config': settings.to_dict(), 'source': metadata,
                'training_scenarios': train_seeds, 'validation_scenarios': val_seeds,
                'completed_epochs': epoch, 'state_dict': copy.deepcopy(model.state_dict())}

    manifest = {'initial_probe.pt': _export(checkpoint(0), out/'initial_probe.pt')}
    rows, best, best_accuracy, updates = [], None, -1., 0
    for epoch in range(settings.epochs+1):
        if epoch:
            model.train()
            indices = torch.randperm(len(training['actions']), generator=sampler)
            for ids in indices.split(settings.batch_size):
                logits, _ = model(training['maps'][ids], training['vector'][ids])
                loss = nn.functional.cross_entropy(logits, training['actions'][ids])
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite imitation loss')
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm,
                                         error_if_nonfinite=True)
                optimizer.step()
                updates += 1
        if epoch % settings.evaluation_interval == 0 or epoch == settings.epochs:
            measured = {}
            for split, data in [('train', training), ('validation', validation)]:
                measured[split] = label_metrics(model, data, settings.batch_size)
                rows.append({'epoch': epoch, 'updates': updates, 'split': split,
                             **measured[split]})
            if epoch and measured['validation']['joint_accuracy'] > best_accuracy:
                best_accuracy = measured['validation']['joint_accuracy']
                best = checkpoint(epoch)
                best['selection'] = {'metric': 'validation joint_accuracy', 'value': best_accuracy}
            atomic_csv_save(rows, out/'label_metrics.csv')
            print(f'Imitation seed {settings.seed}: epoch {epoch}/{settings.epochs}, '
                  f'train agreement {measured["train"]["joint_accuracy"]:.3f}, '
                  f'validation agreement {measured["validation"]["joint_accuracy"]:.3f}',
                  flush=True)
    manifest['final_probe.pt'] = _export(checkpoint(settings.epochs), out/'final_probe.pt')
    manifest['selected_probe.pt'] = _export(best, out/'selected_probe.pt')
    atomic_json_save(manifest, out/'probe_manifest.json')
    summary = {'method': METHOD, 'completed_epochs': settings.epochs, 'updates': updates,
               'best_validation_joint_accuracy': best_accuracy,
               'selected_epoch': best['completed_epochs'], 'probe_config': settings.to_dict(),
               'final_train_metrics': measured['train'],
               'final_validation_metrics': measured['validation']}
    atomic_json_save(summary, out/'probe_summary.json')
    return summary
