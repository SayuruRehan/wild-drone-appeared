"""Small, explicit DQN, Monte Carlo REINFORCE and n-step A2C implementations."""

import copy
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from wild_drone.checkpoint_audit import audit_checkpoints
from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.experiments import run_episode, validate_seed_split, write_metadata
from wild_drone.training_io import (
    atomic_csv_save,
    atomic_json_save,
    atomic_torch_save,
    pack_replay,
    restore_rng,
    rng_state,
    unpack_replay,
    validate_resume,
)


@dataclass(frozen=True)
class TrainConfig:
    episodes: int = 20
    seed: int = 0
    scenario_seed_start: int = 0
    learning_rate: float = 0.0003
    gamma: float = 0.99
    batch_size: int = 32
    replay_size: int = 2000
    warmup_steps: int = 128
    target_update_steps: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10000
    n_steps: int = 16
    entropy_weight: float = 0.01
    value_weight: float = 0.5
    max_grad_norm: float = 0.5
    device: str = 'cpu'
    spatial_encoding: bool = True
    monte_carlo_baseline: bool = True
    double_dqn: bool = False
    checkpoint_interval: int = 25
    progress_interval: int = 10
    validation_interval: int = 25
    validation_seeds: tuple[int, ...] = ()

    def __post_init__(self):
        for name in ('episodes', 'batch_size', 'replay_size', 'warmup_steps',
                     'target_update_steps', 'epsilon_decay_steps', 'n_steps'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.replay_size < self.batch_size:
            raise ValueError('replay_size must be at least batch_size')
        if not 0 <= self.gamma <= 1:
            raise ValueError('gamma must be in [0, 1]')
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError('require 0 <= epsilon_end <= epsilon_start <= 1')
        for name in ('learning_rate', 'max_grad_norm'):
            if not 0 < getattr(self, name) < float('inf'):
                raise ValueError(f'{name} must be finite and positive')
        for name in ('entropy_weight', 'value_weight'):
            if not 0 <= getattr(self, name) < float('inf'):
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.seed < 0 or self.scenario_seed_start < 0:
            raise ValueError('seeds must be nonnegative')
        if self.device not in ('cpu', 'cuda', 'auto'):
            raise ValueError('device must be cpu, cuda or auto')
        for name in ('checkpoint_interval', 'validation_interval'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not isinstance(self.progress_interval, int) or self.progress_interval < 0:
            raise ValueError('progress_interval must be a nonnegative integer')
        for name in ('spatial_encoding', 'monte_carlo_baseline', 'double_dqn'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be a boolean')
        seeds = tuple(self.validation_seeds)
        if len(seeds) != len(set(seeds)) or any(not isinstance(s, int) or s < 0 for s in seeds):
            raise ValueError('validation seeds must be distinct nonnegative integers')
        object.__setattr__(self, 'validation_seeds', seeds)

    def to_dict(self):
        data = asdict(self)
        data['validation_seeds'] = list(self.validation_seeds)
        return data


def resolve_device(requested):
    if requested not in ('cpu', 'cuda', 'auto'):
        raise ValueError('device must be cpu, cuda or auto')
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but no CUDA-capable PyTorch runtime is available')
    if requested == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    return torch.device(requested)


def discounted_returns(rewards, gamma, bootstrap=0.0):
    result = np.empty(len(rewards), dtype=np.float32)
    running = float(bootstrap)
    for i in range(len(rewards)-1, -1, -1):
        running = rewards[i] + gamma*running
        result[i] = running
    return result


def td_targets(rewards, terminals, next_values, gamma):
    return rewards + gamma*(~terminals).float()*next_values


def dqn_next_values(target_q, online_q=None):
    """DQN target max, or Double DQN online selection / target evaluation."""
    if online_q is None:
        return target_q.max(dim=1).values
    actions = online_q.argmax(dim=1, keepdim=True)
    return target_q.gather(1, actions).squeeze(1)


def discounted_policy_loss(log_probs, advantages, gamma, *, start_step, normalizer):
    """Absolute-time weighting; fixed normalisation cannot favour short failures."""
    times = torch.arange(start_step, start_step+len(log_probs), dtype=log_probs.dtype,
                         device=log_probs.device)
    weights = torch.pow(torch.full_like(times, gamma), times)
    return -(log_probs*advantages*weights).sum()/normalizer


def monte_carlo_terms(log_probs, rewards, gamma, *, horizon, values=None):
    """REINFORCE terms: full reward-to-go, no value bootstrap or return centering."""
    returns = torch.from_numpy(discounted_returns(rewards, gamma)).to(log_probs.device)
    advantage = returns if values is None else returns-values.detach()
    actor = discounted_policy_loss(log_probs, advantage, gamma, start_step=0, normalizer=horizon)
    critic = (torch.zeros((), device=log_probs.device) if values is None else
              nn.functional.mse_loss(values, returns))
    return actor, critic


def tensor_observations(observations, *, spatial_encoding=True, device='cpu'):
    if isinstance(observations, dict):
        observations = [observations]
    arrays = np.stack([o['maps'] for o in observations])
    if spatial_encoding:
        n = arrays.shape[-1]
        locations = np.zeros((len(observations), 4, n, n), dtype=np.float32)
        for batch, obs in enumerate(observations):
            for i, drone in enumerate(obs['drones']):
                if drone[3] < 0.5:
                    row, col = np.rint(drone[:2]*(n-1)).astype(int)
                    locations[batch, i, row, col] = 1
            row, col = np.rint(obs['globals'][9:11]*(n-1)).astype(int)
            locations[batch, 3, row, col] = 1
        arrays = np.concatenate([arrays, locations], axis=1)
    maps = torch.from_numpy(arrays)
    vector = torch.from_numpy(np.stack([
        np.concatenate([o['drones'].ravel(), o['globals']]) for o in observations]))
    return maps.to(device), vector.to(device)


class FleetNetwork(nn.Module):
    def __init__(self, grid_size, critic=False, spatial_encoding=True):
        super().__init__()
        self.spatial_encoding = spatial_encoding
        channels = 9 if spatial_encoding else 5
        self.conv = nn.Sequential(nn.Conv2d(channels, 16, 3, stride=2, padding=1), nn.ReLU(),
                                  nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                                  nn.Flatten())
        with torch.no_grad():
            size = self.conv(torch.zeros(1, channels, grid_size, grid_size)).shape[1]
        self.encoder = nn.Sequential(nn.Linear(size+24, 128), nn.ReLU())
        self.head = nn.Linear(128, 125)
        self.critic = nn.Linear(128, 1) if critic else None

    def forward(self, maps, vector):
        features = self.encoder(torch.cat([self.conv(maps), vector], dim=1))
        outputs = self.head(features)
        values = self.critic(features).squeeze(-1) if self.critic is not None else None
        return outputs, values


class NeuralPolicy:
    def __init__(self, model, algorithm, env_config, train_config, device='cpu'):
        self.model = model.eval()
        self.algorithm, self.env_config, self.train_config = algorithm, env_config, train_config
        self.device = resolve_device(device)
        self.model.to(self.device)

    def act(self, env, observation):
        if env.config != self.env_config:
            raise ValueError('checkpoint environment differs from evaluation config')
        with torch.no_grad():
            logits, _ = self.model(*tensor_observations(
                observation, spatial_encoding=self.model.spatial_encoding, device=self.device))
        return int(logits.argmax(dim=1).item())


def load_policy(path, device='cpu'):
    # Only tensors and primitive metadata; weights_only avoids arbitrary pickle execution.
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('action_mask') is not None:
        from wild_drone.masked_learning import load_masked_policy

        return load_masked_policy(path, device=device)
    config = EnvConfig(**data['env_config'])
    settings = data['train_config'].copy()
    if 'model_config' not in data:
        # Version-1 checkpoints used five map channels and no REINFORCE baseline.
        settings.update(spatial_encoding=False, monte_carlo_baseline=False)
    model_config = data.get('model_config', {'critic': data['algorithm'] == 'a2c',
                                            'spatial_encoding': False})
    model = FleetNetwork(config.grid_size, **model_config)
    model.load_state_dict(data['state_dict'])
    return NeuralPolicy(model, data['algorithm'], config, TrainConfig(**settings), device)


def _optimise(optimiser, model, loss, grad_norm):
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite training loss')
    optimiser.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
    optimiser.step()
    return float(loss.detach())


def train(algorithm, env_config, train_config, output_dir, *, resume_from=None):
    """Train from scratch or restore a complete episode boundary (total budget = episodes)."""
    if algorithm not in ('dqn', 'reinforce', 'a2c'):
        raise ValueError(f'unknown learning method: {algorithm}')
    t = train_config
    validate_seed_split(t.scenario_seed_start, t.episodes, t.validation_seeds)
    device = resolve_device(t.device)
    state = None
    if resume_from is not None:
        state = torch.load(resume_from, map_location='cpu', weights_only=True)
        validate_resume(state, algorithm, env_config, t)
    out = Path(output_dir)
    if state is None and any((out/file).exists() for file in ['training_state.pt', 'policy.pt']):
        raise ValueError('existing training state: pass resume_from or choose another output')
    random.seed(t.seed)
    np.random.seed(t.seed)
    torch.manual_seed(t.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(t.seed)
    env = WildfireFleetEnv(env_config)
    model_config = {'critic': algorithm == 'a2c' or (
        algorithm == 'reinforce' and t.monte_carlo_baseline),
        'spatial_encoding': t.spatial_encoding}
    model = FleetNetwork(env_config.grid_size, **model_config).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=t.learning_rate)
    target = FleetNetwork(env_config.grid_size, spatial_encoding=t.spatial_encoding).to(
        device) if algorithm == 'dqn' else None
    if target is not None:
        target.load_state_dict(model.state_dict())
        target.eval()
    replay = deque(maxlen=t.replay_size)
    rows, validation_rows, updates, steps = [], [], 0, 0
    last_loss, last_value_loss = None, None
    best_score, best_checkpoint, completed, prior_seconds = None, None, 0, 0.0
    if state is not None:
        if state['model_config'] != model_config:
            raise ValueError('resume model architecture differs from requested settings')
        model.load_state_dict(state['state_dict'])
        optimiser.load_state_dict(state['optimizer'])
        if target is not None:
            target.load_state_dict(state['target'])
        unpack_replay(state['replay'], replay)
        rows, validation_rows = state['rows'], state['validation_rows']
        updates, steps, completed = state['updates'], state['steps'], state['completed_episodes']
        last_loss, last_value_loss = state['last_loss'], state['last_value_loss']
        best_score, best_checkpoint = state['best_score'], state['best_checkpoint']
        if best_checkpoint is not None:
            # Keep the whole selection experiment's seed range after extending its budget.
            best_checkpoint['train_config'] = t.to_dict()
        prior_seconds = state['elapsed_seconds']
        restore_rng(state['rng'], rng, device)  # Restore after constructors consume randomness.
    out.mkdir(parents=True, exist_ok=True)
    metadata = write_metadata(out, env_config, command='train', algorithm=algorithm,
                              train_config=t.to_dict(), device=str(device),
                              torch_device_name=torch.cuda.get_device_name(device)
                              if device.type == 'cuda' else 'CPU',
                              resume_from=str(resume_from) if resume_from else None)
    source_history = state.get('source_history', []) if state else []
    source_history.append({k: metadata[k] for k in ['git_revision', 'git_dirty', 'python',
                                                   'packages', 'device', 'torch_device_name']} |
                          {'resumed_episode': completed})
    atomic_json_save(metadata | {'source_history': source_history}, out/'metadata.json')
    started = time.perf_counter()

    def tensors(observations):
        return tensor_observations(observations, spatial_encoding=t.spatial_encoding, device=device)

    def policy_checkpoint():
        return {'format_version': 2, 'algorithm': algorithm, 'env_config': env_config.to_dict(),
                'train_config': t.to_dict(), 'model_config': model_config,
                'state_dict': model.state_dict(), 'completed_episodes': completed,
                'source_history': source_history}

    def persist():
        seconds = prior_seconds + time.perf_counter()-started
        summary = {'algorithm': algorithm, 'steps': steps, 'updates': updates,
                   'last_loss': last_loss, 'completed_episodes': completed,
                   'episode_returns': [r['return'] for r in rows], 'device': str(device),
                   'elapsed_seconds': seconds, 'steps_per_second': steps/seconds if seconds else 0,
                   'validation_seeds': list(t.validation_seeds),
                   'best_validation_return': best_score,
                   'env_config': env_config.to_dict(), 'train_config': t.to_dict(),
                   'model_config': model_config}
        checkpoint = policy_checkpoint()
        # Commit authority first; an interrupted first export must remain resumable.
        atomic_torch_save(checkpoint | {
            'optimizer': optimiser.state_dict(), 'target': target.state_dict() if target else None,
            'replay': pack_replay(replay), 'rng': rng_state(rng), 'rows': rows,
            'validation_rows': validation_rows, 'updates': updates, 'steps': steps,
            'last_loss': last_loss, 'last_value_loss': last_value_loss,
            'best_score': best_score, 'best_checkpoint': best_checkpoint,
            'elapsed_seconds': seconds}, out/'training_state.pt')
        atomic_torch_save(checkpoint, out/'policy.pt')
        if best_checkpoint is not None:
            atomic_torch_save(best_checkpoint, out/'best_policy.pt')
        atomic_csv_save(rows, out/'training.csv')
        atomic_csv_save(validation_rows, out/'validation.csv')
        atomic_json_save(summary, out/'training_summary.json')
        return summary

    for episode in range(completed, t.episodes):
        observation, _ = env.reset(seed=t.scenario_seed_start+episode)
        logs, values, entropies, rewards = [], [], [], []
        episode_return, fresh_sum, overdue_sum, done = 0.0, 0.0, 0.0, False
        while not done:
            if algorithm == 'dqn':
                with torch.no_grad():
                    logits, value = model(*tensors(observation))
                epsilon = t.epsilon_start + min(1, steps/t.epsilon_decay_steps)*(
                    t.epsilon_end-t.epsilon_start)
                action = int(rng.integers(125)) if rng.random() < epsilon else int(
                    logits.argmax(dim=1).item())
            else:
                logits, value = model(*tensors(observation))
                distribution = Categorical(logits=logits)
                chosen = distribution.sample()
                action = int(chosen.item())
                logs.append(distribution.log_prob(chosen).squeeze())
                entropies.append(distribution.entropy().squeeze())
                if value is not None:
                    values.append(value.squeeze())
            next_observation, reward, terminal, truncated, info = env.step(action)
            done = terminal or truncated
            steps += 1
            episode_return += reward
            fresh_sum += info['weighted_freshness']
            overdue_sum += info['overdue_fraction']
            rewards.append(reward)
            if algorithm == 'dqn':
                replay.append((observation, action, reward, next_observation, terminal))
                if steps >= t.warmup_steps and len(replay) >= t.batch_size:
                    ids = rng.choice(len(replay), t.batch_size, replace=False)
                    batch = [replay[int(i)] for i in ids]
                    q, _ = model(*tensors([b[0] for b in batch]))
                    actions = torch.tensor([b[1] for b in batch], device=device)
                    selected = q.gather(1, actions[:, None])
                    with torch.no_grad():
                        next_inputs = tensors([b[3] for b in batch])
                        next_q, _ = target(*next_inputs)
                        online_q = model(*next_inputs)[0] if t.double_dqn else None
                        targets = td_targets(torch.tensor([b[2] for b in batch], device=device),
                                             torch.tensor([b[4] for b in batch], device=device),
                                             dqn_next_values(next_q, online_q), t.gamma)
                    loss = nn.functional.smooth_l1_loss(selected.squeeze(1), targets)
                    last_loss = _optimise(optimiser, model, loss, t.max_grad_norm)
                    updates += 1
                if steps % t.target_update_steps == 0:
                    target.load_state_dict(model.state_dict())
            elif algorithm == 'a2c' and (len(rewards) >= t.n_steps or done):
                with torch.no_grad():
                    _, boot = model(*tensors(next_observation))
                    bootstrap = 0.0 if terminal else float(boot.item())
                returns = torch.from_numpy(discounted_returns(rewards, t.gamma, bootstrap)).to(
                    device)
                predictions = torch.stack(values)
                advantage = returns-predictions.detach()
                policy_loss = discounted_policy_loss(
                    torch.stack(logs), advantage, t.gamma,
                    start_step=env.elapsed-len(rewards), normalizer=t.n_steps)
                value_loss = nn.functional.mse_loss(predictions, returns)
                loss = (policy_loss + t.value_weight*value_loss -
                        t.entropy_weight*torch.stack(entropies).mean())
                last_loss = _optimise(optimiser, model, loss, t.max_grad_norm)
                last_value_loss = float(value_loss.detach())
                updates += 1
                logs, values, entropies, rewards = [], [], [], []
            observation = next_observation
        if algorithm == 'reinforce':
            # Full episode Monte Carlo targets; the optional baseline never bootstraps.
            loss, value_loss = monte_carlo_terms(
                torch.stack(logs), rewards, t.gamma, horizon=env_config.horizon,
                values=torch.stack(values) if t.monte_carlo_baseline else None)
            loss += t.value_weight*value_loss - t.entropy_weight*torch.stack(entropies).mean()
            last_loss = _optimise(optimiser, model, loss, t.max_grad_norm)
            last_value_loss = float(value_loss.detach())
            updates += 1
        completed = episode+1
        rows.append({'episode': completed, 'scenario_seed': t.scenario_seed_start+episode,
                     'training_seed': t.seed, 'return': episode_return,
                     'steps': env.elapsed, 'loss': last_loss, 'value_loss': last_value_loss,
                     'updates': updates, 'total_losses': int(env.lost.sum()),
                     'weighted_freshness': fresh_sum/env_config.horizon,
                     'overdue_fraction': (overdue_sum+env_config.horizon-env.elapsed)/
                     env_config.horizon, **env.totals})
        if t.validation_seeds and (completed % t.validation_interval == 0 or completed ==
                                   t.episodes):
            policy = NeuralPolicy(model, algorithm, env_config, t, device=str(device))
            current = [dict(episode=completed, **run_episode(
                env_config, policy, seed, name=algorithm, training_seed=t.seed))
                for seed in t.validation_seeds]
            validation_rows.extend(current)
            score = float(np.mean([r['return'] for r in current]))
            if best_score is None or score > best_score:
                best_score = score
                best_checkpoint = copy.deepcopy(policy_checkpoint())
                best_checkpoint['selection'] = {'validation_return': score,
                                                'validation_seeds': list(t.validation_seeds)}
            model.train()
        if t.progress_interval and completed % t.progress_interval == 0:
            print(f'{algorithm}: episode {completed}/{t.episodes}, return {episode_return:.3f}, '
                  f'updates {updates}, device {device}', flush=True)
        if completed % t.checkpoint_interval == 0 or completed == t.episodes:
            summary = persist()
    # A no-op resume still regenerates portable exports from the authoritative state.
    if state is not None and state['completed_episodes'] == t.episodes:
        summary = persist()
    # Audit once on completion, before callers evaluate or archive portable exports.
    atomic_json_save(audit_checkpoints(out), out/'checkpoint_manifest.json')
    return summary
