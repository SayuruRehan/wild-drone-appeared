"""Small, explicit DQN, Monte Carlo REINFORCE and n-step A2C implementations."""

import csv
import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv


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


def discounted_returns(rewards, gamma, bootstrap=0.0):
    result = np.empty(len(rewards), dtype=np.float32)
    running = float(bootstrap)
    for i in range(len(rewards)-1, -1, -1):
        running = rewards[i] + gamma*running
        result[i] = running
    return result


def td_targets(rewards, terminals, next_values, gamma):
    return rewards + gamma*(~terminals).float()*next_values


def discounted_policy_loss(log_probs, advantages, gamma, *, start_step, normalizer):
    """Absolute-time weighting; fixed normalisation cannot favour short failures."""
    times = torch.arange(start_step, start_step+len(log_probs), dtype=log_probs.dtype,
                         device=log_probs.device)
    weights = torch.pow(torch.full_like(times, gamma), times)
    return -(log_probs*advantages*weights).sum()/normalizer


def tensor_observations(observations):
    if isinstance(observations, dict):
        observations = [observations]
    maps = torch.from_numpy(np.stack([o['maps'] for o in observations]))
    vector = torch.from_numpy(np.stack([
        np.concatenate([o['drones'].ravel(), o['globals']]) for o in observations]))
    return maps, vector


class FleetNetwork(nn.Module):
    def __init__(self, grid_size, critic=False):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(5, 16, 3, stride=2, padding=1), nn.ReLU(),
                                  nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                                  nn.Flatten())
        with torch.no_grad():
            size = self.conv(torch.zeros(1, 5, grid_size, grid_size)).shape[1]
        self.encoder = nn.Sequential(nn.Linear(size+24, 128), nn.ReLU())
        self.head = nn.Linear(128, 125)
        self.critic = nn.Linear(128, 1) if critic else None

    def forward(self, maps, vector):
        features = self.encoder(torch.cat([self.conv(maps), vector], dim=1))
        outputs = self.head(features)
        values = self.critic(features).squeeze(-1) if self.critic is not None else None
        return outputs, values


class NeuralPolicy:
    def __init__(self, model, algorithm, env_config, train_config):
        self.model = model.eval()
        self.algorithm, self.env_config, self.train_config = algorithm, env_config, train_config

    def act(self, env, observation):
        if env.config != self.env_config:
            raise ValueError('checkpoint environment differs from evaluation config')
        with torch.no_grad():
            logits, _ = self.model(*tensor_observations(observation))
        return int(logits.argmax(dim=1).item())


def load_policy(path):
    # Only tensors and primitive metadata; weights_only avoids arbitrary pickle execution.
    data = torch.load(path, map_location='cpu', weights_only=True)
    config = EnvConfig(**data['env_config'])
    model = FleetNetwork(config.grid_size, critic=data['algorithm'] == 'a2c')
    model.load_state_dict(data['state_dict'])
    return NeuralPolicy(model, data['algorithm'], config, TrainConfig(**data['train_config']))


def _optimise(optimiser, model, loss, grad_norm):
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite training loss')
    optimiser.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
    optimiser.step()
    return float(loss.detach())


def train(algorithm, env_config, train_config, output_dir):
    if algorithm not in ('dqn', 'reinforce', 'a2c'):
        raise ValueError(f'unknown learning method: {algorithm}')
    t = train_config
    random.seed(t.seed)
    np.random.seed(t.seed)
    torch.manual_seed(t.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(t.seed)
    env = WildfireFleetEnv(env_config)
    model = FleetNetwork(env_config.grid_size, critic=algorithm == 'a2c')
    optimiser = torch.optim.Adam(model.parameters(), lr=t.learning_rate)
    target = FleetNetwork(env_config.grid_size) if algorithm == 'dqn' else None
    if target is not None:
        target.load_state_dict(model.state_dict())
        target.eval()
    replay = deque(maxlen=t.replay_size)
    rows, updates, steps, last_loss = [], 0, 0, None
    for episode in range(t.episodes):
        observation, _ = env.reset(seed=t.scenario_seed_start+episode)
        logs, values, entropies, rewards = [], [], [], []
        episode_return, done = 0.0, False
        while not done:
            logits, value = model(*tensor_observations(observation))
            if algorithm == 'dqn':
                epsilon = t.epsilon_start + min(1, steps/t.epsilon_decay_steps)*(
                    t.epsilon_end-t.epsilon_start)
                action = int(rng.integers(125)) if rng.random() < epsilon else int(
                    logits.argmax(dim=1).item())
            else:
                distribution = Categorical(logits=logits)
                chosen = distribution.sample()
                action = int(chosen.item())
                logs.append(distribution.log_prob(chosen).squeeze())
                entropies.append(distribution.entropy().squeeze())
                if value is not None:
                    values.append(value.squeeze())
            next_observation, reward, terminal, truncated, _ = env.step(action)
            done = terminal or truncated
            steps += 1
            episode_return += reward
            rewards.append(reward)
            if algorithm == 'dqn':
                replay.append((observation, action, reward, next_observation, terminal))
                if steps >= t.warmup_steps and len(replay) >= t.batch_size:
                    ids = rng.choice(len(replay), t.batch_size, replace=False)
                    batch = [replay[int(i)] for i in ids]
                    q, _ = model(*tensor_observations([b[0] for b in batch]))
                    selected = q.gather(1, torch.tensor([b[1] for b in batch])[:, None])
                    with torch.no_grad():
                        next_q, _ = target(*tensor_observations([b[3] for b in batch]))
                        targets = td_targets(torch.tensor([b[2] for b in batch]),
                                             torch.tensor([b[4] for b in batch]),
                                             next_q.max(dim=1).values, t.gamma)
                    loss = nn.functional.smooth_l1_loss(selected.squeeze(1), targets)
                    last_loss = _optimise(optimiser, model, loss, t.max_grad_norm)
                    updates += 1
                if steps % t.target_update_steps == 0:
                    target.load_state_dict(model.state_dict())
            elif algorithm == 'a2c' and (len(rewards) >= t.n_steps or done):
                with torch.no_grad():
                    _, boot = model(*tensor_observations(next_observation))
                    bootstrap = 0.0 if terminal else float(boot.item())
                returns = torch.from_numpy(discounted_returns(rewards, t.gamma, bootstrap))
                predictions = torch.stack(values)
                advantage = returns-predictions.detach()
                policy_loss = discounted_policy_loss(
                    torch.stack(logs), advantage, t.gamma,
                    start_step=env.elapsed-len(rewards), normalizer=t.n_steps)
                value_loss = nn.functional.mse_loss(predictions, returns)
                loss = (policy_loss + t.value_weight*value_loss -
                        t.entropy_weight*torch.stack(entropies).mean())
                last_loss = _optimise(optimiser, model, loss, t.max_grad_norm)
                updates += 1
                logs, values, entropies, rewards = [], [], [], []
            observation = next_observation
        if algorithm == 'reinforce':
            returns = torch.from_numpy(discounted_returns(rewards, t.gamma))
            # Plain Monte Carlo estimator. No TD critic or per-episode return centering.
            loss = discounted_policy_loss(torch.stack(logs), returns, t.gamma,
                                          start_step=0, normalizer=env_config.horizon)
            loss -= t.entropy_weight*torch.stack(entropies).mean()
            last_loss = _optimise(optimiser, model, loss, t.max_grad_norm)
            updates += 1
        rows.append({'episode': episode+1, 'scenario_seed': t.scenario_seed_start+episode,
                     'training_seed': t.seed, 'return': episode_return,
                     'steps': env.elapsed, 'loss': last_loss, 'updates': updates,
                     'total_losses': int(env.lost.sum())})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = {'algorithm': algorithm, 'env_config': env_config.to_dict(),
                  'train_config': asdict(t), 'state_dict': model.state_dict()}
    torch.save(checkpoint, out/'policy.pt')
    with (out/'training.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'algorithm': algorithm, 'steps': steps, 'updates': updates,
               'last_loss': last_loss, 'episode_returns': [r['return'] for r in rows],
               'env_config': env_config.to_dict(), 'train_config': asdict(t)}
    (out/'training_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary
