"""Double DQN with n-step returns for every model kind, with feasibility-constrained
action selection both when acting and inside the bootstrap target.

    PYTHONPATH=research:src python -m fleetrl.train --kind structured --env dev20 \
        --train-drones 3 --steps 150000 --seed 0 --out research/runs/pilot/structured-0

Episodes draw world seeds from the training split; periodic evaluation uses the first
``eval_worlds`` development worlds and never touches final-test worlds.
"""

import argparse
import json
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fleetrl.env import FleetEnv
from fleetrl.evaluate import run_episode
from fleetrl.features import FeatureSpec, build_features, collate, snapshot
from fleetrl.feasibility import feasible_mask
from fleetrl.models import FleetQ
from wild_drone.config import EnvConfig

RESEARCH = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    kind: str = 'vdn'
    env: str = 'dev20'
    train_drones: tuple = (3,)
    steps: int = 150_000
    seed: int = 0
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    n_step: int = 3
    batch_size: int = 64
    replay_size: int = 100_000
    warmup_steps: int = 5_000
    train_every: int = 4
    target_update_steps: int = 2_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_fraction: float = 0.3
    reward_scale: float = 10.0
    max_grad_norm: float = 10.0
    hidden: int = 128
    local_radius: int = 8
    global_size: int = 8
    eval_every: int = 25_000
    eval_worlds: int = 10
    eval_drones: tuple = (3,)
    device: str = 'auto'
    threads: int = 1
    notes: dict = field(default_factory=dict)


def env_config(name):
    return EnvConfig(**json.load(open(RESEARCH/'configs/environments.json'))[name])


def splits():
    return json.load(open(RESEARCH/'configs/splits.json'))


def make_model(cfg: TrainConfig, spec: FeatureSpec):
    n = cfg.train_drones[0] if cfg.kind == 'joint' else None
    if cfg.kind == 'joint' and len(cfg.train_drones) != 1:
        raise ValueError('the joint model trains on one fleet size')
    return FleetQ(cfg.kind, 2*spec.local_radius+1, spec.global_size, cfg.hidden, n)


class LearnedPolicy:
    """Greedy (argmax) policy over feasible joint actions."""

    def __init__(self, model, spec, device):
        self.model, self.spec, self.device = model, spec, device

    @torch.no_grad()
    def act_commands(self, env):
        f = build_features(collate([snapshot(env)], self.device), self.spec)
        q = self.model.dense_q(f)[0]
        mask = torch.as_tensor(feasible_mask(env), device=self.device)
        index = int(q.masked_fill(~mask, float('-inf')).argmax())
        return np.array(np.unravel_index(index, (5,)*env.n_drones), dtype=np.int64)


def load_policy(run_dir, checkpoint='final.pt', device='cpu'):
    run_dir = Path(run_dir)
    cfg = TrainConfig(**json.load(open(run_dir/'config.json')))
    config = env_config(cfg.env)
    spec = FeatureSpec.from_config(config, local_radius=cfg.local_radius,
                                   global_size=cfg.global_size)
    model = make_model(cfg, spec).to(device)
    model.load_state_dict(torch.load(run_dir/checkpoint, map_location=device))
    model.eval()
    return LearnedPolicy(model, spec, device), cfg, config


def evaluate(policy, config, drones, seeds):
    rows = []
    policy.model.eval()
    for n in drones:
        for s in seeds:
            rows.append(run_episode(config, n, policy, s))
    policy.model.train()
    return rows


def train(cfg: TrainConfig, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.threads)
    device = torch.device('cuda' if cfg.device == 'auto' and torch.cuda.is_available()
                          else ('cpu' if cfg.device == 'auto' else cfg.device))
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    config = env_config(cfg.env)
    spec = FeatureSpec.from_config(config, local_radius=cfg.local_radius,
                                   global_size=cfg.global_size)
    model = make_model(cfg, spec).to(device)
    target = make_model(cfg, spec).to(device)
    target.load_state_dict(model.state_dict())
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    policy = LearnedPolicy(model, spec, device)
    split = splits()
    train_start = split['training']['start'] + cfg.seed*100_000
    dev = split['development']
    eval_seeds = list(range(dev['start'], dev['start']+cfg.eval_worlds))
    if cfg.kind == 'joint':  # a joint head exists only for its training fleet size
        cfg.eval_drones = tuple(n for n in cfg.eval_drones if n in cfg.train_drones) or \
            tuple(cfg.train_drones)
    json.dump(asdict(cfg), open(out/'config.json', 'w'), indent=1)

    replay = {n: deque(maxlen=cfg.replay_size // len(cfg.train_drones))
              for n in cfg.train_drones}
    log = open(out/'log.jsonl', 'a')
    steps, episode, updates = 0, 0, 0
    t0 = time.time()
    best = -np.inf
    ep_returns = deque(maxlen=20)

    def learn():
        sizes = np.array([len(replay[n]) for n in cfg.train_drones], dtype=float)
        n = cfg.train_drones[int(rng.choice(len(sizes), p=sizes/sizes.sum()))]
        buf = replay[n]
        batch = [buf[int(i)] for i in rng.integers(0, len(buf), cfg.batch_size)]
        b = len(batch)
        both = build_features(collate([x[0] for x in batch]+[x[4] for x in batch], device), spec)
        a = torch.as_tensor(np.stack([x[1] for x in batch]), device=device)
        r = torch.as_tensor([x[2] for x in batch], dtype=torch.float32, device=device)
        disc = torch.as_tensor([x[3] for x in batch], dtype=torch.float32, device=device)
        m2 = torch.as_tensor(np.stack([x[5] for x in batch]), device=device)
        parts = model.components(both)
        now = tuple(None if t is None else t[:b] for t in parts)
        with torch.no_grad():
            later = tuple(None if t is None else t[b:].detach() for t in parts)
            online_next = model.dense_from(later, both['pairs']).masked_fill(
                ~m2, float('-inf'))
            best_next = online_next.argmax(1, keepdim=True)
            s2 = {key: (val[b:] if torch.is_tensor(val) else val) for key, val in both.items()}
            next_v = target.dense_q(s2).gather(1, best_next).squeeze(1)
            y = r + disc*next_v
        q = model.gather_from(now, both['pairs'], a)
        loss = nn.functional.smooth_l1_loss(q, y)
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimiser.step()
        return float(loss.detach())

    loss_ema = None
    while steps < cfg.steps:
        n = cfg.train_drones[episode % len(cfg.train_drones)]
        env = FleetEnv(config, n)
        env.reset(train_start+episode)
        pending = deque()
        done, ret = False, 0.0
        snap, mask = snapshot(env), feasible_mask(env)
        while not done:
            frac = min(1.0, steps/(cfg.epsilon_fraction*cfg.steps))
            eps = cfg.epsilon_start + frac*(cfg.epsilon_end-cfg.epsilon_start)
            if rng.random() < eps:
                index = int(rng.choice(np.flatnonzero(mask)))
                commands = np.array(np.unravel_index(index, (5,)*n), dtype=np.int64)
            else:
                commands = policy.act_commands(env)
            reward, done, _ = env.step(commands)
            ret += reward
            snap2 = snapshot(env)
            mask2 = feasible_mask(env) if not done else np.ones(5**n, dtype=bool)
            pending.append((snap, commands, cfg.reward_scale*reward))
            if len(pending) == cfg.n_step or done:
                while pending and (len(pending) == cfg.n_step or done):
                    g = sum(cfg.gamma**t*p[2] for t, p in enumerate(pending))
                    disc = 0.0 if done else cfg.gamma**len(pending)
                    replay[n].append((pending[0][0], pending[0][1], g, disc, snap2, mask2))
                    pending.popleft()
                    if not done:
                        break
            snap, mask = snap2, mask2
            steps += 1
            if steps >= cfg.warmup_steps and steps % cfg.train_every == 0:
                value = learn()
                updates += 1
                loss_ema = value if loss_ema is None else 0.99*loss_ema+0.01*value
            if steps % cfg.target_update_steps == 0:
                target.load_state_dict(model.state_dict())
            if steps % cfg.eval_every == 0 or steps == cfg.steps:
                rows = evaluate(policy, config, cfg.eval_drones, eval_seeds)
                entry = {'steps': steps, 'episodes': episode, 'minutes': (time.time()-t0)/60,
                         'epsilon': eps, 'loss': loss_ema,
                         'train_return': float(np.mean(ep_returns)) if ep_returns else None}
                for k in cfg.eval_drones:
                    sel = [x for x in rows if x['n_drones'] == k]
                    entry[f'dev_freshness_n{k}'] = float(np.mean(
                        [x['weighted_freshness'] for x in sel]))
                    entry[f'dev_losses_n{k}'] = int(sum(x['losses'] for x in sel))
                log.write(json.dumps(entry)+'\n')
                log.flush()
                print(json.dumps(entry), flush=True)
                score = entry[f'dev_freshness_n{cfg.eval_drones[0]}']
                torch.save(model.state_dict(), out/'final.pt')
                if score > best:
                    best = score
                    torch.save(model.state_dict(), out/'best_dev.pt')
            if steps >= cfg.steps:
                break
        ep_returns.append(ret)
        episode += 1
    json.dump({'steps': steps, 'episodes': episode, 'updates': updates,
               'minutes': (time.time()-t0)/60, 'best_dev': best},
              open(out/'done.json', 'w'), indent=1)
    return out


def main():
    p = argparse.ArgumentParser()
    for name, f in TrainConfig.__dataclass_fields__.items():
        if name == 'notes':
            continue
        default = f.default
        flag = '--'+name.replace('_', '-')
        if isinstance(default, tuple):
            p.add_argument(flag, nargs='+', type=int, default=list(default))
        else:
            p.add_argument(flag, type=type(default), default=default)
    p.add_argument('--out', required=True)
    a = vars(p.parse_args())
    out = Path(a.pop('out'))
    for key in ('train_drones', 'eval_drones'):
        a[key] = tuple(a[key])
    train(TrainConfig(**a), out)


if __name__ == '__main__':
    main()
