"""Hypothesis test: RL with the baselines' feasible-action set (action masking).

Reuses the repo's network, observation transform and update formulas; only adds a mask.
Usage: masked_probe.py ALGO EPISODES SEED [--no-mask]
"""
import json
import sys
import time
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from wild_drone.baselines import candidate_actions
from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.learning import (FleetNetwork, discounted_policy_loss, discounted_returns,
                                 monte_carlo_terms, td_targets, tensor_observations)

algo, episodes, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
use_mask = '--no-mask' not in sys.argv
config = EnvConfig(**json.load(open('configs/development.json'))['environment'])
VAL = range(5000, 5005)
import os
REWARD_SCALE = float(os.environ.get('REWARD_SCALE', 0))
lr, gamma, batch, replay_size, warmup, target_every = 3e-4, float(os.environ.get('GAMMA', 0.99)), 32, 2000, 128, 100
eps_decay, n_steps, ent_w, val_w, clip = 10000, 16, 0.01, 0.5, 0.5

torch.manual_seed(seed)
torch.set_num_threads(1)
rng = np.random.default_rng(seed)


def mask_of(env):
    m = np.zeros(125, dtype=bool)
    if use_mask:
        m[candidate_actions(env)[0]] = True
    else:
        m[:] = True
    return m


def masked(logits, mask):
    return logits.masked_fill(~torch.as_tensor(mask).reshape(logits.shape), -1e9)


critic = algo in ('a2c', 'reinforce')
model = FleetNetwork(config.grid_size, critic=critic)
opt = torch.optim.Adam(model.parameters(), lr=lr)
target = FleetNetwork(config.grid_size) if algo == 'dqn' else None
if target:
    target.load_state_dict(model.state_dict())
replay = deque(maxlen=replay_size)


def step_opt(loss):
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()


def evaluate():
    model.eval()
    out = []
    for s in VAL:
        env = WildfireFleetEnv(config)
        obs, _ = env.reset(seed=s)
        total, done = 0.0, False
        while not done:
            with torch.no_grad():
                logits, _ = model(*tensor_observations(obs))
            a = int(masked(logits, mask_of(env)).argmax().item())
            obs, r, done, _, _ = env.step(a)
            total += r
        out.append((total, int(env.lost.sum()), env.totals['charge_completions']))
    model.train()
    return out


steps, t0 = 0, time.time()
for ep in range(episodes):
    env = WildfireFleetEnv(config)
    obs, _ = env.reset(seed=ep)
    mask = mask_of(env)
    logs, values, ents, rewards, done, ret = [], [], [], [], False, 0.0
    while not done:
        if algo == 'dqn':
            eps = 1 + min(1, steps/eps_decay)*(0.05-1)
            if rng.random() < eps:
                a = int(rng.choice(np.flatnonzero(mask)))
            else:
                with torch.no_grad():
                    q, _ = model(*tensor_observations(obs))
                a = int(masked(q, mask).argmax().item())
        else:
            logits, v = model(*tensor_observations(obs))
            dist = Categorical(logits=masked(logits, mask)[0])
            chosen = dist.sample()
            a = int(chosen.item())
            logs.append(dist.log_prob(chosen)); ents.append(dist.entropy()); values.append(v[0])
        nobs, r, term, _, info = env.step(a)
        done = term
        nmask = mask_of(env) if not done else np.ones(125, dtype=bool)
        steps += 1; ret += r
        if REWARD_SCALE:  # training-only reward; evaluation keeps the original return
            r = REWARD_SCALE*info['reward_components']['freshness_gain']
        rewards.append(r)
        if algo == 'dqn':
            replay.append((obs, a, r, nobs, term, nmask))
            if steps >= warmup:
                b = [replay[int(i)] for i in rng.choice(len(replay), batch, replace=False)]
                q, _ = model(*tensor_observations([x[0] for x in b]))
                sel = q.gather(1, torch.tensor([x[1] for x in b])[:, None]).squeeze(1)
                with torch.no_grad():
                    nq, _ = target(*tensor_observations([x[3] for x in b]))
                    nq = masked(nq, np.stack([x[5] for x in b]))
                    tgt = td_targets(torch.tensor([x[2] for x in b], dtype=torch.float32),
                                     torch.tensor([x[4] for x in b]), nq.max(1).values, gamma)
                step_opt(nn.functional.smooth_l1_loss(sel, tgt))
            if steps % target_every == 0:
                target.load_state_dict(model.state_dict())
        elif algo == 'a2c' and (len(rewards) >= n_steps or done):
            with torch.no_grad():
                boot = 0.0 if term else float(model(*tensor_observations(nobs))[1].item())
            rets = torch.from_numpy(discounted_returns(rewards, gamma, boot))
            pred = torch.stack(values)
            pl = discounted_policy_loss(torch.stack(logs), rets-pred.detach(), gamma,
                                        start_step=env.elapsed-len(rewards), normalizer=n_steps)
            step_opt(pl + val_w*nn.functional.mse_loss(pred, rets) - ent_w*torch.stack(ents).mean())
            logs, values, ents, rewards = [], [], [], []
        obs, mask = nobs, nmask
    if algo == 'reinforce':
        pl, vl = monte_carlo_terms(torch.stack(logs), rewards, gamma, horizon=config.horizon,
                                   values=torch.stack(values))
        step_opt(pl + val_w*vl - ent_w*torch.stack(ents).mean())
    if (ep+1) % 25 == 0:
        ev = evaluate()
        print(json.dumps({'algo': algo, 'mask': use_mask, 'gamma': gamma, 'reward_scale': REWARD_SCALE, 'seed': seed, 'episode': ep+1,
                          'train_return': round(ret, 2),
                          'val_return': round(float(np.mean([e[0] for e in ev])), 3),
                          'val_losses': float(np.mean([e[1] for e in ev])),
                          'val_charges': float(np.mean([e[2] for e in ev])),
                          'minutes': round((time.time()-t0)/60, 1)}), flush=True)
