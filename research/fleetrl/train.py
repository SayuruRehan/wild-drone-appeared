"""Double DQN with n-step returns for every model kind, with feasibility-constrained
action selection both when acting and inside the bootstrap target.

    PYTHONPATH=research:src python -m fleetrl.train --kind structured --env dev20 \
        --train-drones 3 --steps 150000 --seed 0 --out research/runs/pilot/structured-0

Episodes draw world seeds from the training split; periodic evaluation uses the first
``eval_worlds`` development worlds and never touches final-test worlds.

Runs are resumable: every ``checkpoint_every`` steps (at the next episode boundary) the
model, target, optimiser, replay buffer, RNG states and counters are written atomically to
``checkpoint.pt``; starting the same command again continues from there (e.g. after a
Colab disconnect). The checkpoint is deleted when the run completes unless
``keep_checkpoint`` is set.
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
from fleetrl.features import FeatureSpec, build_features, collate, collate_arrays, snapshot
from fleetrl.feasibility import feasible_mask
from fleetrl.models import FleetQ
from fleetrl.replay import Replay
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
    model_version: int = 2
    dcg_rank: int = 1
    checkpoint_every: int = 25_000
    keep_checkpoint: bool = False
    notes: dict = field(default_factory=dict)


def env_config(name):
    return EnvConfig(**json.load(open(RESEARCH/'configs/environments.json'))[name])


def splits():
    return json.load(open(RESEARCH/'configs/splits.json'))


def make_model(cfg: TrainConfig, spec: FeatureSpec):
    n = cfg.train_drones[0] if cfg.kind == 'joint' else None
    if cfg.kind == 'joint' and len(cfg.train_drones) != 1:
        raise ValueError('the joint model trains on one fleet size')
    return FleetQ(cfg.kind, 2*spec.local_radius+1, spec.global_size, cfg.hidden, n,
                  model_version=cfg.model_version, dcg_rank=cfg.dcg_rank)


def read_config(run_dir):
    data = json.load(open(Path(run_dir)/'config.json'))
    data.setdefault('model_version', 1)  # pilot runs predate the field
    for key in ('train_drones', 'eval_drones'):
        data[key] = tuple(data[key])
    return TrainConfig(**data)


class LearnedPolicy:
    """Greedy (argmax) policy over feasible joint actions."""

    def __init__(self, model, spec, device):
        self.model, self.spec, self.device = model, spec, device

    guard_trips = 0      # times the network's choice was not feasible (non-finite values)

    @torch.no_grad()
    def act_commands(self, env):
        f = build_features(collate([snapshot(env)], self.device), self.spec)
        q = self.model.dense_q(f)[0]
        mask_np = feasible_mask(env)
        mask = torch.as_tensor(mask_np, device=self.device)
        # NaN/inf values would make argmax ignore the mask; treat them as worst-but-feasible
        q = torch.nan_to_num(q.float(), nan=-1e30, posinf=1e30, neginf=-1e30)
        index = int(q.masked_fill(~mask, float('-inf')).argmax())
        if not mask_np[index]:  # safety net: never execute an infeasible joint action
            LearnedPolicy.guard_trips += 1
            index = int(np.flatnonzero(mask_np)[0])
        return np.array(np.unravel_index(index, (5,)*env.n_drones), dtype=np.int64)


def load_policy(run_dir, checkpoint='final.pt', device='cpu'):
    run_dir = Path(run_dir)
    cfg = read_config(run_dir)
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


def _slice(parts, sl):
    """Slice every tensor in a (possibly nested) tuple of model components."""
    if parts is None:
        return None
    if isinstance(parts, tuple):
        return tuple(_slice(p, sl) for p in parts)
    return parts[sl]


def _atomic_save(obj, path):
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


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
    if cfg.kind == 'joint':  # a joint head exists only for its training fleet size
        cfg.eval_drones = tuple(n for n in cfg.eval_drones if n in cfg.train_drones) or \
            tuple(cfg.train_drones)
    model = make_model(cfg, spec).to(device)
    target = make_model(cfg, spec).to(device)
    target.load_state_dict(model.state_dict())
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    policy = LearnedPolicy(model, spec, device)
    split = splits()
    train_start = split['training']['start'] + cfg.seed*100_000
    dev = split['development']
    eval_seeds = list(range(dev['start'], dev['start']+cfg.eval_worlds))

    replay = {n: Replay(cfg.replay_size // len(cfg.train_drones), n) for n in cfg.train_drones}
    st = {'steps': 0, 'episode': 0, 'updates': 0, 'best': -np.inf, 'loss_ema': None,
          'elapsed': 0.0, 'last_checkpoint': 0, 'ep_returns': []}
    ckpt_path = out/'checkpoint.pt'
    if ckpt_path.exists():
        if json.load(open(out/'config.json')) != json.loads(json.dumps(asdict(cfg))):
            raise ValueError(f'{out} holds a checkpoint from a different configuration')
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        target.load_state_dict(ck['target'])
        optimiser.load_state_dict(ck['optimiser'])
        replay = {int(n): Replay.from_state_dict(d) for n, d in ck['replay'].items()}
        st = ck['state']
        rng.bit_generator.state = ck['numpy_rng']
        random.setstate(ck['python_rng'])
        torch.set_rng_state(ck['torch_rng'])
        if torch.cuda.is_available() and ck.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        # drop log lines written after the checkpoint (they will be re-done)
        if (out/'log.jsonl').exists():
            lines = [line for line in open(out/'log.jsonl')
                     if json.loads(line)['steps'] <= st['steps']]
            open(out/'log.jsonl', 'w').writelines(lines)
        print(json.dumps({'resumed_at_steps': st['steps']}), flush=True)
    else:
        json.dump(asdict(cfg), open(out/'config.json', 'w'), indent=1)
        open(out/'log.jsonl', 'w').close()
    log = open(out/'log.jsonl', 'a')
    t0 = time.time() - st['elapsed']

    def save_checkpoint():
        st['elapsed'] = time.time() - t0
        st['last_checkpoint'] = st['steps']
        _atomic_save({'model': model.state_dict(), 'target': target.state_dict(),
                      'optimiser': optimiser.state_dict(),
                      'replay': {n: r.state_dict() for n, r in replay.items()},
                      'state': st, 'numpy_rng': rng.bit_generator.state,
                      'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state(),
                      'cuda_rng': (torch.cuda.get_rng_state_all()
                                   if torch.cuda.is_available() else None)}, ckpt_path)

    def learn():
        sizes = np.array([len(replay[n]) for n in cfg.train_drones], dtype=float)
        n = cfg.train_drones[int(rng.choice(len(sizes), p=sizes/sizes.sum()))]
        s, a, r, disc, s2, m2 = replay[n].sample(rng, cfg.batch_size)
        b = len(a)
        both = build_features(collate_arrays(
            {k: np.concatenate([s[k], s2[k]]) for k in s}, device), spec)
        a = torch.as_tensor(a, device=device)
        r = torch.as_tensor(r, device=device)
        disc = torch.as_tensor(disc, device=device)
        m2 = torch.as_tensor(m2, device=device)
        parts = model.components(both)
        with torch.no_grad():
            later = _slice(parts, slice(b, None))
            online_next = model.dense_from(later, both['pairs']).masked_fill(
                ~m2, float('-inf'))
            best_next = online_next.argmax(1, keepdim=True)
            s2f = {key: (val[b:] if torch.is_tensor(val) else val) for key, val in both.items()}
            next_v = target.dense_q(s2f).gather(1, best_next).squeeze(1)
            y = r + disc*next_v
        q = model.gather_from(_slice(parts, slice(None, b)), both['pairs'], a)
        loss = nn.functional.smooth_l1_loss(q, y)
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimiser.step()
        return float(loss.detach())

    while st['steps'] < cfg.steps:
        episode = st['episode']
        n = cfg.train_drones[episode % len(cfg.train_drones)]
        buf = replay[n]
        env = FleetEnv(config, n)
        env.reset(train_start+episode)
        pending = deque()
        done, ret = False, 0.0
        snap, mask = snapshot(env), feasible_mask(env)
        sid = buf.add_state(snap)
        while not done:
            steps = st['steps']
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
            sid2 = buf.add_state(snap2)
            mask2 = feasible_mask(env) if not done else np.ones(5**n, dtype=bool)
            pending.append((sid, commands, cfg.reward_scale*reward))
            while pending and (len(pending) == cfg.n_step or done):
                g = sum(cfg.gamma**t*p[2] for t, p in enumerate(pending))
                disc = 0.0 if done else cfg.gamma**len(pending)
                buf.add(pending[0][0], pending[0][1], g, disc, sid2, mask2)
                pending.popleft()
                if not done:
                    break
            sid, mask = sid2, mask2
            st['steps'] = steps = steps+1
            if steps >= cfg.warmup_steps and steps % cfg.train_every == 0:
                value = learn()
                st['updates'] += 1
                st['loss_ema'] = (value if st['loss_ema'] is None
                                  else 0.99*st['loss_ema']+0.01*value)
            if steps % cfg.target_update_steps == 0:
                target.load_state_dict(model.state_dict())
            if steps % cfg.eval_every == 0 or steps == cfg.steps:
                rows = evaluate(policy, config, cfg.eval_drones, eval_seeds)
                recent = st['ep_returns'][-20:]
                entry = {'steps': steps, 'episodes': episode,
                         'minutes': (time.time()-t0)/60, 'epsilon': eps,
                         'loss': st['loss_ema'],
                         'train_return': float(np.mean(recent)) if recent else None}
                for k in cfg.eval_drones:
                    sel = [x for x in rows if x['n_drones'] == k]
                    entry[f'dev_freshness_n{k}'] = float(np.mean(
                        [x['weighted_freshness'] for x in sel]))
                    entry[f'dev_losses_n{k}'] = int(sum(x['losses'] for x in sel))
                entry['guard_trips'] = LearnedPolicy.guard_trips
                log.write(json.dumps(entry)+'\n')
                log.flush()
                print(json.dumps(entry), flush=True)
                score = entry[f'dev_freshness_n{cfg.eval_drones[0]}']
                _atomic_save(model.state_dict(), out/'final.pt')
                # keep every evaluated model (~1 MB) for post-hoc analysis, e.g. of losses
                (out/'snapshots').mkdir(exist_ok=True)
                _atomic_save(model.state_dict(), out/'snapshots'/f'step-{steps:07d}.pt')
                if score > st['best']:
                    st['best'] = score
                    _atomic_save(model.state_dict(), out/'best_dev.pt')
            if steps >= cfg.steps:
                break
        st['ep_returns'] = (st['ep_returns'] + [ret])[-20:]
        st['episode'] += 1
        if (st['steps'] - st['last_checkpoint'] >= cfg.checkpoint_every
                and st['steps'] < cfg.steps):
            save_checkpoint()
    json.dump({'steps': st['steps'], 'episodes': st['episode'], 'updates': st['updates'],
               'minutes': (time.time()-t0)/60, 'best_dev': st['best']},
              open(out/'done.json', 'w'), indent=1)
    if ckpt_path.exists() and not cfg.keep_checkpoint:
        ckpt_path.unlink()
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
        elif isinstance(default, bool):
            p.add_argument(flag, type=lambda v: v.lower() in ('1', 'true', 'yes'),
                           default=default)
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
