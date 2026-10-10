"""How large are immediate reward differences between feasible actions, vs return noise?"""
import copy
import json

import numpy as np

from wild_drone.baselines import candidate_actions
from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv

config = EnvConfig(**json.load(open('configs/development.json'))['environment'])
rng = np.random.default_rng(0)
spreads, gaps, stds = [], [], []
for seed in range(5):
    env = WildfireFleetEnv(config)
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        ids, _ = candidate_actions(env)
        if env.elapsed % 10 == 0 and len(ids) > 1:
            rewards = []
            for a in ids:
                sim = copy.deepcopy(env)
                rewards.append(sim.step(int(a))[1])
            rewards = np.sort(rewards)
            spreads.append(rewards[-1]-rewards[0])
            gaps.append(rewards[-1]-rewards[-2])
            stds.append(np.std(rewards))
        obs, r, done, _, _ = env.step(int(rng.choice(ids)))
print(f'immediate reward across feasible actions: best-worst {np.mean(spreads):.4f}, '
      f'best-second {np.mean(gaps):.5f}, std {np.mean(stds):.4f}')

# Return noise under one fixed (shielded-random) behaviour across scenarios and repeats.
returns = []
for seed in range(20):
    env = WildfireFleetEnv(config)
    env.reset(seed=seed)
    done, total = False, 0.0
    while not done:
        _, r, done, _, _ = env.step(int(rng.choice(candidate_actions(env)[0])))
        total += r
    returns.append(total)
print(f'episode return std under masked random: {np.std(returns):.3f} (mean {np.mean(returns):.2f})')
