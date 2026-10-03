"""Conventional controllers, with explicit battery-aware heuristics."""

import copy

import numpy as np

from wild_drone.environment import MOVES, decode_action

COMMANDS = np.stack([decode_action(a) for a in range(125)])


def candidate_actions(env):
    """Feasible baseline proposals. This is NOT a shield applied to RL policies."""
    c = env.config
    proposed = env.positions[None, :, :] + MOVES[COMMANDS]
    valid = ~np.any((proposed < 0) | (proposed >= c.grid_size), axis=(1, 2))
    distances = np.abs(proposed-env.base).sum(axis=2)
    current_distances = np.abs(env.positions-env.base).sum(axis=1)
    for i in range(3):
        if env.lost[i]:
            valid &= COMMANDS[:, i] == 0
            continue
        if current_distances[i] == 0 and env.batteries[i] < c.battery_capacity:
            valid &= COMMANDS[:, i] == 0
            continue
        costs = np.where(COMMANDS[:, i] == 0, c.hover_cost, c.move_cost)
        if current_distances[i] == 0:
            costs[COMMANDS[:, i] == 0] = 0
        low_battery = env.batteries[i] <= current_distances[i]*c.move_cost + 3*c.move_cost
        if low_battery and current_distances[i] > 0:
            valid &= distances[:, i] < current_distances[i]
        else:
            valid &= ((env.batteries[i]-costs >= distances[:, i]*c.move_cost+c.move_cost)
                      | (distances[:, i] == 0))
    for i in range(3):
        for j in range(i+1, 3):
            if env.lost[i] or env.lost[j]:
                continue
            same = np.all(proposed[:, i] == proposed[:, j], axis=1)
            base = np.all(proposed[:, i] == env.base, axis=1)
            swap = (np.all(proposed[:, i] == env.positions[j], axis=1) &
                    np.all(proposed[:, j] == env.positions[i], axis=1))
            if np.array_equal(env.positions[i], env.positions[j]):
                swap[:] = False
            valid &= ~(same & ~base) & ~swap
    indices = np.flatnonzero(valid)
    # If resource/collision constraints make every heuristic proposal impossible,
    # do not silently teleport or restore battery; the environment still decides.
    if not len(indices):
        indices = np.array([0])
    return indices, proposed[indices]


class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, observation):
        return int(self.rng.integers(125))


class GreedyPolicy:
    def act(self, env, observation):
        ids, positions = candidate_actions(env)
        r, col = np.indices((env.config.grid_size, env.config.grid_size))
        covered = ((np.abs(r[None, None]-positions[:, :, 0, None, None]) <=
                    env.config.sensor_radius) &
                   (np.abs(col[None, None]-positions[:, :, 1, None, None]) <=
                    env.config.sensor_radius) & ~env.lost[None, :, None, None]).any(axis=1)
        urgency = env.priority_weights()*np.minimum(
            (env.ages+1)/env.config.inspection_deadline, 1)
        score = (covered*urgency).sum(axis=(1, 2))
        # Equal-information candidates favour less movement; no future reward estimate.
        score -= np.count_nonzero(COMMANDS[ids], axis=1)*0.001
        return int(ids[np.argmax(score)])


class SweepPolicy:
    def __init__(self):
        self.waypoints = None
        self.cursors = np.zeros(3, dtype=int)

    def _initialise(self, env):
        n, radius = env.config.grid_size, env.config.sensor_radius
        self.waypoints = []
        for i, columns in enumerate(np.array_split(np.arange(n), 3)):
            points = []
            rows = list(range(min(radius, n-1), n, max(1, 2*radius+1)))
            if rows[-1] != n-1:
                rows.append(n-1)
            for k, row in enumerate(rows):
                ends = [int(columns[0]), int(columns[-1])]
                if k % 2:
                    ends.reverse()
                points.extend((row, col) for col in ends)
            self.waypoints.append(np.array(points))
            self.cursors[i] = 0

    def act(self, env, observation):
        if self.waypoints is None or env.elapsed == 0:
            self._initialise(env)
        targets = []
        for i in range(3):
            points = self.waypoints[i]
            target = points[self.cursors[i]]
            if np.array_equal(env.positions[i], target):
                self.cursors[i] = (self.cursors[i]+1) % len(points)
                target = points[self.cursors[i]]
            distance = int(np.abs(env.positions[i]-env.base).sum())
            if env.batteries[i] <= distance*env.config.move_cost+3*env.config.move_cost:
                target = env.base
            targets.append(target)
        ids, proposed = candidate_actions(env)
        score = -np.abs(proposed-np.array(targets)[None]).sum(axis=(1, 2))
        return int(ids[np.argmax(score)])


class RollingHorizonPolicy:
    """Bounded stochastic shooting planner; independent hypothetical futures."""

    def __init__(self, seed=0, horizon=4, candidates=8, samples=2, gamma=0.99):
        if min(horizon, candidates, samples) < 1:
            raise ValueError('planner horizon, candidates and samples must be positive')
        self.rng = np.random.default_rng(seed)
        self.horizon, self.candidates, self.samples, self.gamma = (
            horizon, candidates, samples, gamma)
        self.greedy = GreedyPolicy()

    def act(self, env, observation):
        valid, _ = candidate_actions(env)
        first = [self.greedy.act(env, observation)]
        first.extend(int(a) for a in self.rng.choice(valid, self.candidates-1))
        future_seeds = self.rng.integers(0, 2**31, self.samples)
        proposal_seeds = self.rng.integers(0, 2**31, len(first))
        scores = []
        for action, proposal_seed in zip(first, proposal_seeds, strict=True):
            returns = []
            for future_seed in future_seeds:
                model = copy.deepcopy(env)
                # Do not use the cloned real RNG state to predict what happens next.
                model.np_random = np.random.default_rng(int(future_seed))
                proposal_rng = np.random.default_rng(int(proposal_seed))
                total, command = 0.0, action
                for t in range(self.horizon):
                    obs, reward, done, _, _ = model.step(command)
                    total += self.gamma**t*reward
                    if done:
                        break
                    if proposal_rng.random() < 0.75:
                        command = self.greedy.act(model, obs)
                    else:
                        choices, _ = candidate_actions(model)
                        command = int(proposal_rng.choice(choices))
                total += 0.1*model._metrics()['weighted_freshness']
                returns.append(total)
            scores.append(np.mean(returns))
        return first[int(np.argmax(scores))]


def make_policy(name, seed=0, **planner_kwargs):
    if name == 'random':
        return RandomPolicy(seed)
    if name == 'sweep':
        return SweepPolicy()
    if name == 'greedy':
        return GreedyPolicy()
    if name == 'planner':
        return RollingHorizonPolicy(seed, **planner_kwargs)
    raise ValueError(f'unknown baseline: {name}')
