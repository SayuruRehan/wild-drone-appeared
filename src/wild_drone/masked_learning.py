"""DQN, REINFORCE and A2C restricted to the baselines' feasible-action set.

Every conventional controller proposes actions through `baselines.candidate_actions`, which
removes off-grid moves, collisions and moves that leave too little battery to return. The
original learners have no such rule and lose the fleet. Here the same rule masks the learned
policy: infeasible joint actions get zero probability (actors) or are excluded from the
argmax and the TD target (DQN). The mask is a deterministic function of the complete state;
the environment, reward and transitions are unchanged. This is a shield in the sense of
Alshiekh et al. (2018), shared with the baselines, not a safety guarantee proven here.

Network, observation transform and update formulas are imported from `learning`, so the
only difference from the unmasked methods is action selection and the DQN target.
"""

import copy
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from wild_drone.baselines import COMMANDS, candidate_actions
from wild_drone.config import EnvConfig
from wild_drone.environment import MOVES, WildfireFleetEnv
from wild_drone.experiments import run_episode, validate_seed_split, write_metadata, write_rows
from wild_drone.learning import (
    FleetNetwork,
    discounted_policy_loss,
    discounted_returns,
    monte_carlo_terms,
    resolve_device,
    td_targets,
    tensor_observations,
)
from wild_drone.training_io import atomic_json_save, atomic_torch_save

MASK_RULE = 'baseline-candidate-actions'


def action_mask(env):
    """Boolean (125,) mask of joint actions the baselines' proposal rule allows.

    The rule checks battery and collisions jointly. When low-battery drones' only base-ward
    cells collide, no joint action passes and it falls back to all-stay, so the drones hover
    until a battery empties (observed with a learned policy). In that case only, use
    `emergency_return_mask` instead.
    """
    mask = np.zeros(125, dtype=bool)
    mask[candidate_actions(env)[0]] = True
    away = ~env.lost & np.any(env.positions != env.base, axis=1)
    distances = np.abs(env.positions-env.base).sum(axis=1)
    low = away & (env.batteries <= (distances+3)*env.config.move_cost)
    if mask.sum() == 1 and mask[0] and low.any():
        return emergency_return_mask(env)
    return mask


def emergency_return_mask(env):
    """Collision-free return: each away drone moves closer to base or waits.

    A blocked move still costs a full move, so conflicting proposals are excluded. Drones
    with the smallest battery margin (battery minus distance) get priority to move; only
    the joint actions with the best priority-weighted number of moving drones remain.
    """
    positions, base = env.positions, env.base
    away = ~env.lost & np.any(positions != base, axis=1)
    distances = np.abs(positions-base).sum(axis=1)
    proposed = positions[None] + MOVES[COMMANDS]
    closer = np.abs(proposed-base).sum(axis=2) < distances
    feasible = np.all(np.where(away, closer | (COMMANDS == 0), COMMANDS == 0), axis=1)
    for i in range(3):
        for j in range(i+1, 3):
            if env.lost[i] or env.lost[j]:
                continue
            same = np.all(proposed[:, i] == proposed[:, j], axis=1)
            at_base = np.all(proposed[:, i] == base, axis=1)
            swap = (np.all(proposed[:, i] == positions[j], axis=1) &
                    np.all(proposed[:, j] == positions[i], axis=1))
            feasible &= ~(same & ~at_base) & ~swap
    margin = np.where(away, env.batteries-distances*env.config.move_cost, np.inf)
    priority = np.zeros(3)
    priority[np.argsort(margin, kind='stable')] = [4, 2, 1]  # critical drone outranks others
    score = ((COMMANDS != 0) & away) @ priority
    score[~feasible] = -1
    return score == score.max()


def apply_mask(logits, mask):
    """Give infeasible actions the lowest finite logit: zero probability, finite entropy."""
    mask = torch.as_tensor(mask, device=logits.device).reshape(logits.shape)
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def masked_max(q_values, masks):
    """Maximum Q over feasible next actions only (DQN target)."""
    return apply_mask(q_values, masks).max(dim=1).values


class MaskedRandomPolicy:
    """Uniform over feasible joint actions: the reference a masked learner must beat."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, observation):
        return int(self.rng.choice(np.flatnonzero(action_mask(env))))


class MaskedNeuralPolicy:
    def __init__(self, model, algorithm, env_config, device='cpu'):
        self.model = model.eval()
        self.algorithm, self.env_config = algorithm, env_config
        self.device = resolve_device(device)
        self.model.to(self.device)

    def act(self, env, observation):
        if env.config != self.env_config:
            raise ValueError('checkpoint environment differs from evaluation config')
        with torch.no_grad():
            logits, _ = self.model(*tensor_observations(
                observation, spatial_encoding=self.model.spatial_encoding, device=self.device))
        return int(apply_mask(logits, action_mask(env)).argmax(dim=1).item())


def load_masked_policy(path, device='cpu'):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('action_mask') != MASK_RULE:
        raise ValueError('not a masked-policy checkpoint')
    config = EnvConfig(**data['env_config'])
    model = FleetNetwork(config.grid_size, **data['model_config'])
    model.load_state_dict(data['state_dict'])
    return MaskedNeuralPolicy(model, data['algorithm'], config, device)


def train_masked(algorithm, env_config, train_config, output_dir):
    """Train one masked learner from scratch (no resume); returns the summary dict."""
    if algorithm not in ('dqn', 'reinforce', 'a2c'):
        raise ValueError(f'unknown learning method: {algorithm}')
    t = train_config
    validate_seed_split(t.scenario_seed_start, t.episodes, t.validation_seeds)
    out = Path(output_dir)
    if (out/'policy.pt').exists():
        raise ValueError(f'existing masked policy in {out}; choose another output folder')
    device = resolve_device(t.device)
    random.seed(t.seed)
    np.random.seed(t.seed)
    torch.manual_seed(t.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(t.seed)
    env = WildfireFleetEnv(env_config)
    model_config = {'critic': algorithm == 'a2c' or (
        algorithm == 'reinforce' and t.monte_carlo_baseline),
        'spatial_encoding': t.spatial_encoding}
    model = FleetNetwork(env_config.grid_size, **model_config).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=t.learning_rate)
    target = None
    if algorithm == 'dqn':
        target = FleetNetwork(env_config.grid_size, spatial_encoding=t.spatial_encoding).to(device)
        target.load_state_dict(model.state_dict())
        target.eval()
    replay = deque(maxlen=t.replay_size)
    rows, validation_rows, updates, steps, last_loss = [], [], 0, 0, None
    best_score, best_checkpoint = None, None
    metadata = write_metadata(out, env_config, command='train_masked', algorithm=algorithm,
                              train_config=t.to_dict(), action_mask=MASK_RULE,
                              device=str(device))
    started = time.perf_counter()

    def tensors(observations):
        return tensor_observations(observations, spatial_encoding=t.spatial_encoding, device=device)

    def optimise(loss):
        nonlocal last_loss, updates
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite training loss')
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), t.max_grad_norm)
        optimiser.step()
        last_loss, updates = float(loss.detach()), updates+1

    def checkpoint(completed):
        return {'format_version': 2, 'action_mask': MASK_RULE, 'algorithm': algorithm,
                'env_config': env_config.to_dict(), 'train_config': t.to_dict(),
                'model_config': model_config, 'state_dict': copy.deepcopy(model.state_dict()),
                'completed_episodes': completed, 'git_revision': metadata['git_revision']}

    for episode in range(t.episodes):
        observation, _ = env.reset(seed=t.scenario_seed_start+episode)
        mask = action_mask(env)
        logs, values, entropies, rewards = [], [], [], []
        episode_return, fresh_sum, overdue_sum, done = 0.0, 0.0, 0.0, False
        while not done:
            if algorithm == 'dqn':
                epsilon = t.epsilon_start + min(1, steps/t.epsilon_decay_steps)*(
                    t.epsilon_end-t.epsilon_start)
                if rng.random() < epsilon:
                    action = int(rng.choice(np.flatnonzero(mask)))
                else:
                    with torch.no_grad():
                        q, _ = model(*tensors(observation))
                    action = int(apply_mask(q, mask).argmax(dim=1).item())
            else:
                logits, value = model(*tensors(observation))
                distribution = Categorical(logits=apply_mask(logits, mask))
                chosen = distribution.sample()
                action = int(chosen.item())
                logs.append(distribution.log_prob(chosen).squeeze())
                entropies.append(distribution.entropy().squeeze())
                if value is not None:
                    values.append(value.squeeze())
            next_observation, reward, terminal, truncated, info = env.step(action)
            done = terminal or truncated
            # Terminal next states are never bootstrapped, so their mask is irrelevant.
            next_mask = np.ones(125, dtype=bool) if done else action_mask(env)
            steps += 1
            episode_return += reward
            fresh_sum += info['weighted_freshness']
            overdue_sum += info['overdue_fraction']
            rewards.append(reward)
            if algorithm == 'dqn':
                replay.append((observation, action, reward, next_observation, terminal, next_mask))
                if steps >= t.warmup_steps and len(replay) >= t.batch_size:
                    batch = [replay[int(i)] for i in rng.choice(len(replay), t.batch_size,
                                                                 replace=False)]
                    q, _ = model(*tensors([b[0] for b in batch]))
                    actions = torch.tensor([b[1] for b in batch], device=device)
                    selected = q.gather(1, actions[:, None]).squeeze(1)
                    with torch.no_grad():
                        next_q, _ = target(*tensors([b[3] for b in batch]))
                        targets = td_targets(
                            torch.tensor([b[2] for b in batch], device=device),
                            torch.tensor([b[4] for b in batch], device=device),
                            masked_max(next_q, np.stack([b[5] for b in batch])), t.gamma)
                    optimise(nn.functional.smooth_l1_loss(selected, targets))
                if steps % t.target_update_steps == 0:
                    target.load_state_dict(model.state_dict())
            elif algorithm == 'a2c' and (len(rewards) >= t.n_steps or done):
                with torch.no_grad():
                    _, boot = model(*tensors(next_observation))
                bootstrap = 0.0 if terminal else float(boot.item())
                returns = torch.from_numpy(discounted_returns(rewards, t.gamma, bootstrap)).to(
                    device)
                predictions = torch.stack(values)
                policy_loss = discounted_policy_loss(
                    torch.stack(logs), returns-predictions.detach(), t.gamma,
                    start_step=env.elapsed-len(rewards), normalizer=t.n_steps)
                optimise(policy_loss + t.value_weight*nn.functional.mse_loss(predictions, returns)
                         - t.entropy_weight*torch.stack(entropies).mean())
                logs, values, entropies, rewards = [], [], [], []
            observation, mask = next_observation, next_mask
        if algorithm == 'reinforce':
            actor, critic = monte_carlo_terms(
                torch.stack(logs), rewards, t.gamma, horizon=env_config.horizon,
                values=torch.stack(values) if t.monte_carlo_baseline else None)
            optimise(actor + t.value_weight*critic - t.entropy_weight*torch.stack(entropies).mean())
        completed = episode+1
        rows.append({'episode': completed, 'scenario_seed': t.scenario_seed_start+episode,
                     'training_seed': t.seed, 'return': episode_return, 'steps': env.elapsed,
                     'loss': last_loss, 'updates': updates, 'total_losses': int(env.lost.sum()),
                     'weighted_freshness': fresh_sum/env_config.horizon,
                     'overdue_fraction': (overdue_sum+env_config.horizon-env.elapsed)/
                     env_config.horizon, **env.totals})
        if t.validation_seeds and (completed % t.validation_interval == 0 or
                                   completed == t.episodes):
            policy = MaskedNeuralPolicy(model, algorithm, env_config, device=str(device))
            current = [dict(episode=completed, **run_episode(
                env_config, policy, seed, name=f'masked_{algorithm}', training_seed=t.seed))
                for seed in t.validation_seeds]
            validation_rows.extend(current)
            score = float(np.mean([r['return'] for r in current]))
            if best_score is None or score > best_score:
                best_score, best_checkpoint = score, checkpoint(completed)
                best_checkpoint['selection'] = {'validation_return': score,
                                                'validation_seeds': list(t.validation_seeds)}
            model.train()
        if t.progress_interval and completed % t.progress_interval == 0:
            print(f'masked {algorithm}: episode {completed}/{t.episodes}, '
                  f'return {episode_return:.3f}, losses {int(env.lost.sum())}', flush=True)
    atomic_torch_save(checkpoint(t.episodes), out/'policy.pt')
    if best_checkpoint is not None:
        atomic_torch_save(best_checkpoint, out/'best_policy.pt')
    write_rows(out/'training.csv', rows)
    if validation_rows:
        write_rows(out/'validation.csv', validation_rows)
    seconds = time.perf_counter()-started
    summary = {'algorithm': algorithm, 'action_mask': MASK_RULE, 'steps': steps,
               'updates': updates, 'last_loss': last_loss, 'completed_episodes': t.episodes,
               'elapsed_seconds': seconds, 'best_validation_return': best_score,
               'best_episode': best_checkpoint['completed_episodes'] if best_checkpoint else None,
               'env_config': env_config.to_dict(), 'train_config': t.to_dict(),
               'model_config': model_config}
    atomic_json_save(summary, out/'training_summary.json')
    return summary
