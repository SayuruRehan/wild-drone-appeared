"""Supervised inspection capability probe; outputs are scores, never RL Q values."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from wild_drone.baselines import make_policy
from wild_drone.learning import FleetNetwork, tensor_observations
from wild_drone.masked_learning import MaskedRandomPolicy, action_mask
from wild_drone.ranking_probe import gain_scores, transition_coverage
from wild_drone.training_io import atomic_torch_save

FORMAT = 'wild-drone-inspection-target-v1'


def validate_worlds(training, heldout):
    for seeds in (training, heldout):
        if not seeds or len(set(seeds)) != len(seeds) or any(
                not isinstance(s, int) or s < 0 for s in seeds):
            raise ValueError('world IDs must be distinct nonnegative integers')
    if set(training) & set(heldout):
        raise ValueError('training and held-out world IDs overlap')


def normalised_targets(gains, masks):
    gains, masks = np.asarray(gains), np.asarray(masks)
    if (gains.ndim != 2 or gains.shape[1] != 125 or masks.shape != gains.shape or
            masks.dtype != bool or not masks.any(axis=1).all() or
            not np.isfinite(gains).all()):
        raise ValueError('finite Nx125 gains and nonempty boolean masks required')
    low = np.where(masks, gains, np.inf).min(axis=1)
    high = np.where(masks, gains, -np.inf).max(axis=1)
    informative = high-low > 1e-12
    scale = np.where(informative, high-low, 1)
    targets = np.where(masks & informative[:, None], (gains-low[:, None])/scale[:, None], 0)
    return targets.astype(np.float32), informative


def masked_target_loss(scores, targets, masks):
    if scores.shape != targets.shape or masks.shape != scores.shape or not masks.any(1).all():
        raise ValueError('matching nonempty per-state feasible masks required')
    return (((scores-targets)**2*masks).sum(1)/masks.sum(1)).mean()


def state_data(env, coverage=None):
    observation = env._observation()
    maps, vectors = tensor_observations(observation)
    gains = gain_scores(env, transition_coverage(env) if coverage is None else coverage)
    mask = action_mask(env)
    return maps[0], vectors[0], gains, mask, hashlib.sha256(json.dumps(
        {k: v.tolist() for k, v in observation.items()}, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def pack_states(states):
    if not states:
        raise ValueError('collection yielded no states')
    gains, masks = np.stack([s[2] for s in states]), np.stack([s[3] for s in states])
    targets, informative = normalised_targets(gains, masks)
    return {'maps': torch.stack([s[0] for s in states]),
            'vectors': torch.stack([s[1] for s in states]),
            'gains': torch.from_numpy(gains), 'masks': torch.from_numpy(masks),
            'targets': torch.from_numpy(targets), 'informative': torch.from_numpy(informative)}


def collect_states(config, worlds, steps):
    if (not steps or tuple(sorted(set(steps))) != tuple(steps) or
            any(not isinstance(s, int) or not 0 <= s < config.horizon for s in steps)):
        raise ValueError('snapshot steps must be increasing distinct active elapsed times')
    if len(set(worlds)) != len(worlds) or any(not isinstance(w, int) or w < 0 for w in worlds):
        raise ValueError('world IDs must be distinct nonnegative integers')
    from wild_drone.environment import WildfireFleetEnv
    states, records, missing = [], [], []
    for world in worlds:
        for collector in ('masked_random', 'greedy'):
            env = WildfireFleetEnv(config)
            observation, _ = env.reset(seed=world)
            controller = MaskedRandomPolicy(world) if collector == 'masked_random' else \
                make_policy('greedy')
            for step in range(max(steps)+1):
                if step in steps:
                    state = state_data(env)
                    states.append(state)
                    records.append({'case': f'{world}-{collector}-{step}',
                                    'scenario_seed': world, 'collector': collector, 'step': step,
                                    'observation_sha256': state[4]})
                if step == max(steps):
                    break
                observation, _, terminal, truncated, _ = env.step(controller.act(env, observation))
                if terminal or truncated:
                    missing.extend({'scenario_seed': world, 'collector': collector, 'step': s,
                                    'reason': 'collector terminated before snapshot'}
                                   for s in steps if s > step)
                    break
    return pack_states(states), records, missing


def score_model(model, data):
    with torch.no_grad():
        scores = torch.cat([model(data['maps'][i:i+64], data['vectors'][i:i+64])[0]
                            for i in range(0, len(data['maps']), 64)])
    if scores.shape != (len(data['maps']), 125) or not torch.isfinite(scores).all():
        raise ValueError('finite 125-action model scores required')
    return scores.numpy()


def load_target_model(path, config):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('format') != FORMAT or saved.get('env_config') != config.to_dict():
        raise ValueError('matching explicitly supervised inspection checkpoint required')
    model = FleetNetwork(config.grid_size)
    model.load_state_dict(saved['state_dict'])
    return model.eval()


def fit_targets(config, data, *, seed, updates, batch_size, output):
    if (not isinstance(seed, int) or seed < 0 or not isinstance(updates, int) or updates < 1 or
            not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError('nonnegative seed and positive fitting budget required')
    ids = torch.where(data['informative'])[0]
    if not len(ids):
        raise ValueError('no informative training states')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('use a fresh fitting output directory')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = FleetNetwork(config.grid_size)
    initial = copy.deepcopy(model.state_dict())
    sampler = torch.Generator().manual_seed(seed+1000)
    optimiser = torch.optim.Adam(model.parameters(), lr=.001)
    history, identities = [], {}

    def export(label, completed):
        path = output/f'{label}.pt'
        saved = {'format': FORMAT, 'method': 'supervised-all-feasible-inspection-regression',
                 'env_config': config.to_dict(), 'seed': seed, 'completed_updates': completed,
                 'state_dict': model.state_dict()}
        atomic_torch_save(saved, path)
        loaded = load_target_model(path, config)
        if any(not torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items()):
            raise ValueError('exported inspection model differs')
        identities[path.name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'bytes': path.stat().st_size, 'completed_updates': completed}

    export('initial', 0)
    for update in range(1, updates+1):
        batch = ids[torch.randint(len(ids), (batch_size,), generator=sampler)]
        scores = model(data['maps'][batch], data['vectors'][batch])[0]
        loss = masked_target_loss(scores, data['targets'][batch], data['masks'][batch])
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite inspection loss')
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
        optimiser.step()
        history.append({'update': update, 'training_batch_loss': float(loss.detach())})
    export('final', updates)
    return model.eval(), {'seed': seed, 'completed_updates': updates,
                          'checkpoints': identities}, history, initial
