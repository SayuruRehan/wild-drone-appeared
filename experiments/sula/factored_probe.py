"""Hypothesis 3: drone-centred, parameter-shared, value-decomposed (VDN) masked DQN.

Same environment, reward, mask, gamma, learning rate, replay size, batch and exploration
schedule as masked_probe.py; only the network/observation transform and the joint Q form change.
Usage: factored_probe.py EPISODES SEED
"""
import json
import os
import sys
import time
from collections import deque

import numpy as np
import torch
from torch import nn

from wild_drone.baselines import COMMANDS, candidate_actions
from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.learning import td_targets

episodes, seed = int(sys.argv[1]), int(sys.argv[2])
config = EnvConfig(**json.load(open('configs/development.json'))['environment'])
N = config.grid_size
VAL = range(5000, 5005)
REWARD_SCALE = float(os.environ.get('REWARD_SCALE', 0))
lr, gamma = 3e-4, float(os.environ.get('GAMMA', 0.99))
batch, replay_size, warmup, target_every, eps_decay, clip = 32, 2000, 128, 100, 10000, 0.5
torch.manual_seed(seed)
torch.set_num_threads(1)
rng = np.random.default_rng(seed)
C_ONE_HOT = torch.eye(5)
SIZE = 2*N-1  # drone-centred window covering the whole grid from any position


def features(env):
    """Per-drone egocentric maps (3, 6, SIZE, SIZE) and vectors (3, 8), from full state."""
    c = env.config
    weights = env.priority_weights()
    urgency = weights*np.minimum(env.ages/c.inspection_deadline, 1)/weights.max()
    occupancy = np.zeros((3, N, N), dtype=np.float32)
    for i, (r, col) in enumerate(env.positions):
        if not env.lost[i]:
            occupancy[i, r, col] = 1
    base = np.zeros((N, N), dtype=np.float32)
    base[tuple(env.base)] = 1
    maps = np.zeros((3, 6, SIZE, SIZE), dtype=np.float32)
    vecs = np.zeros((3, 8), dtype=np.float32)
    for i, (r, col) in enumerate(env.positions):
        others = occupancy.sum(0)-occupancy[i]
        stack = np.stack([urgency, env.burning, env.priority/3, others, base,
                          np.ones((N, N))]).astype(np.float32)  # last channel marks in-grid cells
        r0, c0 = N-1-r, N-1-col
        maps[i, :, r0:r0+N, c0:c0+N] = stack
        distance = np.abs(env.positions[i]-env.base).sum()
        vecs[i] = [env.batteries[i]/c.battery_capacity, (env.base[0]-r)/N, (env.base[1]-col)/N,
                   distance*c.move_cost/max(env.batteries[i], 1e-6)/4, 1-env.elapsed/c.horizon,
                   float(env.lost[i]), float(env.charger_owner == i), float(env.charger_owner < 0)]
    return maps, vecs


class DroneQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(6, 16, 3, stride=2, padding=1), nn.ReLU(),
                                  nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                                  nn.Flatten())
        with torch.no_grad():
            size = self.conv(torch.zeros(1, 6, SIZE, SIZE)).shape[1]
        self.mlp = nn.Sequential(nn.Linear(size+8, 128), nn.ReLU(), nn.Linear(128, 5))

    def forward(self, maps, vecs):  # (B, 3, ...) -> per-drone Q (B, 3, 5)
        b = maps.shape[0]
        x = torch.cat([self.conv(maps.flatten(0, 1)), vecs.flatten(0, 1)], dim=1)
        return self.mlp(x).view(b, 3, 5)


CMD = torch.as_tensor(COMMANDS, dtype=torch.long)  # (125, 3)


def joint_q(per_drone):  # (B, 3, 5) -> (B, 125): VDN sum over drones
    return sum(per_drone[:, i, CMD[:, i]] for i in range(3))


def mask_of(env):
    m = np.zeros(125, dtype=bool)
    m[candidate_actions(env)[0]] = True
    return m


def batch_tensors(feats):
    return (torch.from_numpy(np.stack([f[0] for f in feats])),
            torch.from_numpy(np.stack([f[1] for f in feats])))


model, target = DroneQ(), DroneQ()
target.load_state_dict(model.state_dict())
opt = torch.optim.Adam(model.parameters(), lr=lr)
replay = deque(maxlen=replay_size)


def greedy_action(env, feats, mask):
    with torch.no_grad():
        q = joint_q(model(*batch_tensors([feats])))[0]
    return int(q.masked_fill(~torch.as_tensor(mask), -1e9).argmax())


def evaluate():
    out = []
    for s in VAL:
        env = WildfireFleetEnv(config)
        env.reset(seed=s)
        total, done = 0.0, False
        while not done:
            _, r, done, _, _ = env.step(greedy_action(env, features(env), mask_of(env)))
            total += r
        out.append((total, int(env.lost.sum()), env.totals['charge_completions'],
                    env._metrics()['weighted_freshness']))
    return out


steps, t0 = 0, time.time()
for ep in range(episodes):
    env = WildfireFleetEnv(config)
    env.reset(seed=ep)
    feats, mask, done, ret = features(env), mask_of(env), False, 0.0
    while not done:
        eps = 1 + min(1, steps/eps_decay)*(0.05-1)
        a = (int(rng.choice(np.flatnonzero(mask))) if rng.random() < eps
             else greedy_action(env, feats, mask))
        _, r, term, _, info = env.step(a)
        done = term
        nfeats = features(env)
        nmask = mask_of(env) if not done else np.ones(125, dtype=bool)
        steps += 1
        ret += r
        if REWARD_SCALE:  # training-only reward; evaluation keeps the original return
            r = REWARD_SCALE*info['reward_components']['freshness_gain']
        replay.append((feats, a, r, nfeats, term, nmask))
        if steps >= warmup:
            b = [replay[int(i)] for i in rng.choice(len(replay), batch, replace=False)]
            q = joint_q(model(*batch_tensors([x[0] for x in b])))
            sel = q.gather(1, torch.tensor([x[1] for x in b])[:, None]).squeeze(1)
            with torch.no_grad():
                nq = joint_q(target(*batch_tensors([x[3] for x in b])))
                nq = nq.masked_fill(~torch.as_tensor(np.stack([x[5] for x in b])), -1e9)
                tgt = td_targets(torch.tensor([x[2] for x in b], dtype=torch.float32),
                                 torch.tensor([x[4] for x in b]), nq.max(1).values, gamma)
            loss = nn.functional.smooth_l1_loss(sel, tgt)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        if steps % target_every == 0:
            target.load_state_dict(model.state_dict())
        feats, mask = nfeats, nmask
    if (ep+1) % 25 == 0:
        ev = evaluate()
        print(json.dumps({'algo': 'vdn-dqn', 'gamma': gamma, 'reward_scale': REWARD_SCALE, 'seed': seed, 'episode': ep+1,
                          'train_return': round(ret, 2),
                          'val_return': round(float(np.mean([e[0] for e in ev])), 3),
                          'val_freshness': round(float(np.mean([e[3] for e in ev])), 3),
                          'val_losses': float(np.mean([e[1] for e in ev])),
                          'val_charges': float(np.mean([e[2] for e in ev])),
                          'minutes': round((time.time()-t0)/60, 1)}), flush=True)
