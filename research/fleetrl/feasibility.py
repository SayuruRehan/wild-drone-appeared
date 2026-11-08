"""Feasibility of joint actions, factored into per-drone and per-pair constraints.

``wild_drone.baselines.candidate_actions`` (the frozen assignment mask) is a conjunction of

* unary rules: grid bounds, lost drones hover, a drone on the pad that is not full must
  stay and charge, the battery-return rule;
* pairwise rules: no two drones in the same non-base cell, no head-on swaps.

This module computes the two parts separately, so the joint feasible set is
``prod_i allowed[i] AND prod_{i<j} not conflict[i, j]`` for any fleet size. For three
drones the dense set equals the frozen mask exactly (tested).

Joint index convention: C-order over per-drone commands, i.e.
``index = sum_i a_i * 5**(N-1-i)``, which matches ``wild_drone.environment.encode_action``.
"""

import numpy as np

from fleetrl.env import MOVES


def unary_allowed(env):
    """Boolean array (N, 5): which commands each drone may take on its own."""
    c, k = env.config, env.n_drones
    proposed = env.positions[:, None, :] + MOVES[None]                   # (N, 5, 2)
    in_grid = ~np.any((proposed < 0) | (proposed >= c.grid_size), axis=2)
    distances = np.abs(proposed-env.base).sum(axis=2)                     # (N, 5)
    current = np.abs(env.positions-env.base).sum(axis=1)                  # (N,)
    allowed = np.zeros((k, 5), dtype=bool)
    stay_only = np.array([True, False, False, False, False])
    for i in range(k):
        if env.lost[i]:
            allowed[i] = stay_only
            continue
        if current[i] == 0 and env.batteries[i] < c.battery_capacity:
            allowed[i] = stay_only & in_grid[i]
            continue
        costs = np.where(np.arange(5) == 0, c.hover_cost, c.move_cost)
        if current[i] == 0:
            costs[0] = 0
        low_battery = env.batteries[i] <= current[i]*c.move_cost + 3*c.move_cost
        if low_battery and current[i] > 0:
            ok = distances[i] < current[i]
        else:
            ok = ((env.batteries[i]-costs >= distances[i]*c.move_cost+c.move_cost) |
                  (distances[i] == 0))
        allowed[i] = ok & in_grid[i]
    return allowed


def pairwise_conflicts(env):
    """Boolean array (N, N, 5, 5); ``[i, j, a, b]`` is True when drone i taking a and
    drone j taking b is forbidden. Only i < j entries are filled (others False)."""
    k = env.n_drones
    proposed = env.positions[:, None, :] + MOVES[None]                    # (N, 5, 2)
    at_base = np.all(proposed == env.base, axis=2)                        # (N, 5)
    out = np.zeros((k, k, 5, 5), dtype=bool)
    for i in range(k):
        for j in range(i+1, k):
            if env.lost[i] or env.lost[j]:
                continue
            if np.abs(env.positions[i]-env.positions[j]).sum() > 2:
                continue  # cannot meet or swap in one step
            same = np.all(proposed[i][:, None] == proposed[j][None], axis=2)
            swap = (np.all(proposed[i] == env.positions[j], axis=1)[:, None] &
                    np.all(proposed[j] == env.positions[i], axis=1)[None])
            if np.array_equal(env.positions[i], env.positions[j]):
                swap[:] = False
            out[i, j] = (same & ~at_base[i][:, None]) | swap
    return out


def dense_feasible(allowed, conflicts):
    """Dense boolean joint mask of shape (5,)*N from unary and pairwise parts."""
    k = allowed.shape[0]
    mask = np.ones((5,)*k, dtype=bool)
    for i in range(k):
        shape = [1]*k
        shape[i] = 5
        mask &= allowed[i].reshape(shape)
    for i in range(k):
        for j in range(i+1, k):
            if conflicts[i, j].any():
                shape = [1]*k
                shape[i] = shape[j] = 5
                mask &= ~conflicts[i, j].reshape(shape)
    return mask


def feasible_mask(env):
    """Flat boolean mask over all 5**N joint actions (same fallback as the frozen code:
    if nothing is feasible, 'everyone stays' is the only proposal)."""
    mask = dense_feasible(unary_allowed(env), pairwise_conflicts(env)).reshape(-1)
    if not mask.any():
        mask[0] = True
    return mask


def decode(index, n_drones):
    """Joint index -> per-drone commands (C-order)."""
    return np.array(np.unravel_index(index, (5,)*n_drones), dtype=np.int64).reshape(-1)


def encode(commands):
    commands = np.asarray(commands)
    return int(np.ravel_multi_index(tuple(commands), (5,)*len(commands)))
