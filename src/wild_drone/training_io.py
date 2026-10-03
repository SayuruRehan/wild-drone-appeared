"""Atomic episode-boundary checkpoints with portable tensor/RNG state."""

import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def atomic_torch_save(data, path):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    torch.save(data, temporary)
    os.replace(temporary, path)


def atomic_json_save(data, path):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    os.replace(temporary, path)


def atomic_csv_save(rows, path):
    if not rows:
        return
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def pack_replay(replay):
    if not replay:
        return None
    return {
        'observations': {k: torch.from_numpy(np.stack([b[0][k] for b in replay]))
                         for k in replay[0][0]},
        'next_observations': {k: torch.from_numpy(np.stack([b[3][k] for b in replay]))
                              for k in replay[0][3]},
        'actions': torch.tensor([b[1] for b in replay]),
        'rewards': torch.tensor([b[2] for b in replay], dtype=torch.float64),
        'terminals': torch.tensor([b[4] for b in replay]),
    }


def unpack_replay(data, replay):
    if data is None:
        return
    for i in range(len(data['actions'])):
        obs = {k: v[i].numpy().copy() for k, v in data['observations'].items()}
        next_obs = {k: v[i].numpy().copy() for k, v in data['next_observations'].items()}
        replay.append((obs, int(data['actions'][i]), float(data['rewards'][i]), next_obs,
                       bool(data['terminals'][i])))


def rng_state(rng):
    state = np.random.get_state()
    return {
        'sampler': rng.bit_generator.state,
        'python': random.getstate(),
        'numpy': {'name': state[0], 'keys': torch.tensor(state[1].astype(np.int64)),
                  'position': state[2], 'has_gauss': state[3], 'cached_gaussian': state[4]},
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(data, rng, device):
    rng.bit_generator.state = data['sampler']
    random.setstate(data['python'])
    state = data['numpy']
    np.random.set_state((state['name'], state['keys'].numpy().astype(np.uint32),
                         state['position'], state['has_gauss'], state['cached_gaussian']))
    torch.set_rng_state(data['torch'])
    if device.type == 'cuda' and data['cuda']:
        torch.cuda.set_rng_state_all(data['cuda'])


def validate_resume(state, algorithm, environment, settings):
    if state.get('format_version') != 2 or 'optimizer' not in state:
        raise ValueError('resume requires a version-2 training_state.pt, not a policy file')
    if state['algorithm'] != algorithm or state['env_config'] != environment.to_dict():
        raise ValueError('resume algorithm/environment differs from saved conditions')
    allowed_changes = {'episodes', 'device', 'checkpoint_interval', 'progress_interval'}
    current = settings.to_dict()
    # Existing version-2 states predate this opt-in algorithm variant.
    saved = {'double_dqn': False} | state['train_config']
    differences = [key for key in sorted(set(saved) | set(current))
                   if key not in allowed_changes and current.get(key) != saved.get(key)]
    if differences:
        raise ValueError(f'resume learning settings differ: {differences}')
    if settings.episodes < state['completed_episodes']:
        raise ValueError('resume total episodes is less than already completed episodes')
