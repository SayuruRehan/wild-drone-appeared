"""Reward landscape on the 20x20 development config: what simple behaviours score."""
import json
import sys
from collections import defaultdict

import numpy as np

from wild_drone.baselines import COMMANDS, GreedyPolicy, RandomPolicy, SweepPolicy, candidate_actions
from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv

cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/development.json'
overrides = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
config = EnvConfig(**(json.load(open(cfg_path))['environment'] | overrides))
SEEDS = range(5000, 5005)


class Stay:
    def act(self, env, obs):
        return 0


class UnshieldedGreedy:
    """Greedy's scoring over all 125 actions, without the battery/collision filter."""
    def act(self, env, obs):
        n, rad = env.config.grid_size, env.config.sensor_radius
        pos = np.clip(env.positions[None] + np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])[COMMANDS], 0, n-1)
        r, c = np.indices((n, n))
        cov = ((np.abs(r[None, None]-pos[:, :, 0, None, None]) <= rad) &
               (np.abs(c[None, None]-pos[:, :, 1, None, None]) <= rad) & ~env.lost[None, :, None, None]).any(1)
        urg = env.priority_weights()*np.minimum((env.ages+1)/env.config.inspection_deadline, 1)
        return int(np.argmax((cov*urg).sum((1, 2)) - np.count_nonzero(COMMANDS, 1)*0.001))


class ShieldedRandom:
    def __init__(self):
        self.rng = np.random.default_rng(0)

    def act(self, env, obs):
        ids, _ = candidate_actions(env)
        return int(self.rng.choice(ids))


policies = {'stay': Stay, 'random': lambda: RandomPolicy(0), 'shielded_random': ShieldedRandom,
            'sweep': SweepPolicy, 'greedy': GreedyPolicy, 'unshielded_greedy': UnshieldedGreedy}
print(config)
for name, make in policies.items():
    comps, rets, losses, steps, deaths = defaultdict(list), [], [], [], []
    for seed in SEEDS:
        env, pol = WildfireFleetEnv(config), make()
        obs, _ = env.reset(seed=seed)
        total, done, acc, first_loss = 0.0, False, defaultdict(float), None
        while not done:
            obs, r, done, _, info = env.step(pol.act(env, obs))
            total += r
            for k, v in info['reward_components'].items():
                acc[k] += v
            if first_loss is None and info['new_losses']:
                first_loss = env.elapsed
        rets.append(total); losses.append(env.lost.sum()); steps.append(env.elapsed); deaths.append(first_loss)
        for k, v in acc.items():
            comps[k].append(v)
    parts = ' '.join(f'{k}={np.mean(v):+.2f}' for k, v in comps.items())
    print(f'{name:18s} return={np.mean(rets):+7.2f}±{np.std(rets):.2f} losses={np.mean(losses):.1f} '
          f'len={np.mean(steps):.0f} first_loss_step={deaths} | {parts}')
