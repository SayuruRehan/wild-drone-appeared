"""Non-learning controllers for any fleet size.

All controllers act through the same factored feasibility mask as the learners, so
none of them can lose a drone through a battery or collision rule violation.

* ``GreedyPolicy``: the frozen one-step Greedy rule (maximise urgency-weighted union
  coverage next step, tiny penalty per moving drone). Exact enumeration for N <= 5,
  identical to ``wild_drone.baselines.GreedyPolicy`` at N=3 (tested). For N > 5 the
  fleet is split into groups whose footprints/moves cannot interact this step; each
  group of at most 5 drones is solved exactly, larger groups by sequential best response
  followed by coordinate ascent (``exact`` is reported in ``last_exact``).
* ``MaskedRandomPolicy``: uniform over feasible joint actions.
* ``AssignPolicy``: a non-myopic target-assignment heuristic, used to test whether there
  is headroom above Greedy: each drone is assigned a distinct high-urgency region that it
  can reach and return from, and moves toward it.
"""

import numpy as np

from fleetrl.env import MOVES
from fleetrl.feasibility import (
    dense_feasible,
    feasible_mask,
    pairwise_conflicts,
    unary_allowed,
)
from fleetrl.selection import select

EXACT_GREEDY_DRONES = 5


def urgency_map(env):
    """Urgency the inspection reward will pay next step (ages increment before sensing)."""
    return env.priority_weights()*np.minimum((env.ages+1)/env.config.inspection_deadline, 1)


def command_footprints(env):
    """Boolean (N, 5, n, n): cells each drone would cover after each command.
    Out-of-grid moves leave the drone in place, matching the environment."""
    c, k = env.config, env.n_drones
    n, radius = c.grid_size, c.sensor_radius
    proposed = env.positions[:, None, :] + MOVES[None]
    bad = np.any((proposed < 0) | (proposed >= n), axis=2)
    proposed = np.where(bad[..., None], env.positions[:, None, :], proposed)
    out = np.zeros((k, 5, n, n), dtype=bool)
    for i in range(k):
        if env.lost[i]:
            continue
        for a in range(5):
            r, col = proposed[i, a]
            out[i, a, max(0, r-radius):min(n, r+radius+1),
                max(0, col-radius):min(n, col+radius+1)] = True
    return out


class GreedyPolicy:
    def __init__(self):
        self.last_exact = True

    def _exact(self, drones, urgency, foot, allowed, conflicts):
        """Exact greedy over a subset of drones; returns their commands."""
        k = len(drones)
        sub_conf = conflicts[np.ix_(drones, drones)]
        mask = dense_feasible(allowed[drones], sub_conf).reshape(-1)
        if not mask.any():
            return np.zeros(k, dtype=np.int64)
        ids = np.flatnonzero(mask)
        commands = np.array(np.unravel_index(ids, (5,)*k)).T            # (A, k)
        covered = np.zeros((len(ids),)+urgency.shape, dtype=bool)
        for m, i in enumerate(drones):
            covered |= foot[i][commands[:, m]]
        score = (covered*urgency).sum(axis=(1, 2))
        score -= np.count_nonzero(commands, axis=1)*0.001
        return commands[int(np.argmax(score))]

    def _approximate(self, drones, urgency, foot, allowed, conflicts):
        k = len(drones)
        choice = np.zeros(k, dtype=np.int64)

        def ok(m, a, current):
            if not allowed[drones[m], a]:
                return False
            for q in range(k):
                if q == m:
                    continue
                i, j = drones[m], drones[q]
                if i < j and conflicts[i, j, a, current[q]]:
                    return False
                if j < i and conflicts[j, i, current[q], a]:
                    return False
            return True

        def total(current):
            covered = np.zeros(urgency.shape, dtype=bool)
            for m, i in enumerate(drones):
                covered |= foot[i, current[m]]
            return float((covered*urgency).sum()) - np.count_nonzero(current)*0.001

        assigned = np.full(k, -1)
        for m in range(k):  # sequential best response; unassigned drones treated as staying
            current = np.where(assigned < 0, 0, assigned)
            best, best_a = -np.inf, 0
            for a in range(5):
                trial = current.copy()
                trial[m] = a
                if ok(m, a, np.where(assigned < 0, 0, trial)):
                    value = total(trial)
                    if value > best:
                        best, best_a = value, a
            assigned[m] = best_a
        choice = assigned.astype(np.int64)
        improved = True
        while improved:  # coordinate ascent
            improved = False
            base = total(choice)
            for m in range(k):
                for a in range(5):
                    if a == choice[m] or not ok(m, a, choice):
                        continue
                    trial = choice.copy()
                    trial[m] = a
                    value = total(trial)
                    if value > base + 1e-12:
                        choice, base, improved = trial, value, True
        return choice

    def act_commands(self, env):
        k = env.n_drones
        urgency = urgency_map(env)
        foot = command_footprints(env)
        allowed = unary_allowed(env)
        conflicts = pairwise_conflicts(env)
        if k <= EXACT_GREEDY_DRONES:
            mask = dense_feasible(allowed, conflicts).reshape(-1)
            self.last_exact = True
            if not mask.any():
                return np.zeros(k, dtype=np.int64)
            return self._exact(list(range(k)), urgency, foot, allowed, conflicts)
        # Interaction groups: footprints can overlap or drones can collide this step.
        reach = 2*env.config.sensor_radius + 2
        parent = list(range(k))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i in range(k):
            for j in range(i+1, k):
                if np.abs(env.positions[i]-env.positions[j]).max() <= reach:
                    parent[find(i)] = find(j)
        groups = {}
        for i in range(k):
            groups.setdefault(find(i), []).append(i)
        out = np.zeros(k, dtype=np.int64)
        self.last_exact = True
        for drones in groups.values():
            if len(drones) <= EXACT_GREEDY_DRONES:
                out[drones] = self._exact(drones, urgency, foot, allowed, conflicts)
            else:
                self.last_exact = False
                out[drones] = self._approximate(drones, urgency, foot, allowed, conflicts)
        return out


class MaskedRandomPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act_commands(self, env):
        index = int(self.rng.choice(np.flatnonzero(feasible_mask(env))))
        return np.array(np.unravel_index(index, (5,)*env.n_drones), dtype=np.int64)


def box_sum(values, radius):
    """Sum of ``values`` over the (2r+1)^2 box centred at every cell (clipped to grid)."""
    n = values.shape[0]
    integral = np.zeros((n+1, n+1))
    integral[1:, 1:] = values.cumsum(0).cumsum(1)
    lo = np.clip(np.arange(n)-radius, 0, n)
    hi = np.clip(np.arange(n)+radius+1, 0, n)
    return (integral[hi][:, hi] - integral[lo][:, hi] - integral[hi][:, lo] +
            integral[lo][:, lo])


class AssignPolicy:
    """Target assignment with suppression; re-plans every step with hysteresis."""

    def __init__(self, keep=0.8, margin_moves=4):
        self.keep, self.margin = keep, margin_moves
        self.targets = None

    def act_commands(self, env):
        c, k, n = env.config, env.n_drones, env.config.grid_size
        if self.targets is None or env.elapsed == 0 or len(self.targets) != k:
            self.targets = [None]*k
        value = box_sum(urgency_map(env), c.sensor_radius)
        rows, cols = np.indices((n, n))
        home = np.abs(rows-env.base[0]) + np.abs(cols-env.base[1])
        allowed = unary_allowed(env)
        conflicts = pairwise_conflicts(env)
        mask = dense_feasible(allowed, conflicts).reshape(-1)
        if not mask.any():
            return np.zeros(k, dtype=np.int64)
        free = value.copy()
        active = [i for i in range(k) if not env.lost[i] and allowed[i, 1:].any()]
        # drones choose in order of how much they stand to gain
        candidates = {}
        for i in active:
            d = np.abs(rows-env.positions[i][0]) + np.abs(cols-env.positions[i][1])
            reachable = env.batteries[i] - d*c.move_cost >= (home+self.margin)*c.move_cost
            candidates[i] = (d, reachable)
        order = sorted(active, key=lambda i: -np.max(
            np.where(candidates[i][1], value/(candidates[i][0]+2*c.sensor_radius+1), 0)))
        suppress = 2*c.sensor_radius+1
        for i in order:
            d, reachable = candidates[i]
            score = np.where(reachable, free/(d+2*c.sensor_radius+1), -1.0)
            best = np.unravel_index(int(np.argmax(score)), score.shape)
            old = self.targets[i]
            if old is not None and score[old] >= self.keep*score[best] and score[old] > 0:
                best = old
            self.targets[i] = best
            r, col = best
            free[max(0, r-suppress):min(n, r+suppress+1),
                 max(0, col-suppress):min(n, col+suppress+1)] = 0
        foot = command_footprints(env)
        urgency = urgency_map(env)
        norm = max(float(urgency.sum()), 1e-9)
        unary = np.zeros((k, 5))
        for i in range(k):
            gain = (foot[i]*urgency).sum(axis=(1, 2))/norm
            target = self.targets[i] if i in active else None
            if target is None:
                unary[i] = gain
                continue
            proposed = env.positions[i] + MOVES
            dist = np.abs(proposed-np.array(target)).sum(axis=1)
            unary[i] = -dist + 0.5*gain
        index = select(unary, None, mask)
        return np.array(np.unravel_index(index, (5,)*k), dtype=np.int64)


def make_baseline(name, seed=0):
    if name == 'greedy':
        return GreedyPolicy()
    if name == 'masked_random':
        return MaskedRandomPolicy(seed)
    if name == 'assign':
        return AssignPolicy()
    raise ValueError(f'unknown baseline {name!r}')
