"""N-drone generalisation of ``wild_drone.environment.WildfireFleetEnv``.

With ``n_drones=3`` this environment reproduces the frozen assignment environment
exactly: same random draws in the same order, same movement, collision, charging,
reward and metric rules (``research/tests/test_env_equivalence.py`` checks this).
The only interface change is the action: an integer array of ``n_drones`` per-drone
commands (0 stay, 1 up, 2 down, 3 left, 4 right) instead of one joint index in [0, 124].

There is still exactly one charger at the base, so adding drones increases charger
contention; that is one of the interactions this study is about.
"""

import numpy as np

from wild_drone.config import EnvConfig
from wild_drone.environment import MOVES, WIND, neighbour_count

__all__ = ['FleetEnv', 'MOVES', 'WIND']


class FleetEnv:
    def __init__(self, config: EnvConfig | None = None, n_drones: int = 3):
        if not isinstance(n_drones, int) or n_drones < 1:
            raise ValueError('n_drones must be a positive integer')
        self.config = config or EnvConfig()
        self.n_drones = n_drones
        self.age_limit = max(self.config.horizon, self.config.inspection_deadline+1)
        self._done = True

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int):
        c, n, k = self.config, self.config.grid_size, self.n_drones
        self.np_random = rng = np.random.default_rng(seed)
        self.base = np.array([n // 2, n // 2], dtype=np.int32)
        self.terrain = rng.uniform(0.4, 1, (n, n)).astype(np.float32)
        self.fuel = rng.integers(max(1, c.max_fuel // 2), c.max_fuel + 1, (n, n))
        self.fuel[tuple(self.base)] = 0
        self.priority = np.ones((n, n), dtype=np.float32)
        radius = max(1, n // 8)
        for r, col in rng.integers(0, n, size=(3, 2)):
            self.priority[max(0, r-radius):min(n, r+radius+1),
                          max(0, col-radius):min(n, col+radius+1)] = 3
        self.burning = np.zeros((n, n), dtype=bool)
        available = np.flatnonzero(self.fuel)
        self.burning.flat[rng.choice(available, c.initial_fires, replace=False)] = True
        self.ages = np.full((n, n), c.inspection_deadline, dtype=np.int32)
        self.positions = np.tile(self.base, (k, 1))
        self.batteries = np.full(k, c.battery_capacity, dtype=np.float32)
        self.lost = np.zeros(k, dtype=bool)
        self.wind = int(rng.integers(0, 5))
        self.charger_owner = -1
        self.elapsed = 0
        self.totals = dict(energy_used=0.0, collision_attempts=0, safe_returns=0,
                           charge_completions=0, inspected_cells=0, episode_return=0.0,
                           duplicate_cells=0, charger_wait_steps=0)
        self._done = False
        return self._info()

    # ---------------------------------------------------------------- metrics
    def priority_weights(self):
        frontier = (neighbour_count(self.burning) > 0) & (self.fuel > 0) & ~self.burning
        return self.priority * (1 + 2*self.burning + frontier)

    def _metrics(self, weights=None):
        weights = self.priority_weights() if weights is None else weights
        denominator = float(weights.sum())
        freshness = 1-np.minimum(self.ages/self.config.inspection_deadline, 1)
        return {'weighted_freshness': float((weights*freshness).sum()/denominator),
                'overdue_fraction': float((weights*(self.ages >
                                                   self.config.inspection_deadline)).sum()
                                          / denominator),
                'burning_cells': int(self.burning.sum()),
                'total_losses': int(self.lost.sum())}

    def _info(self, **extra):
        return {'step': self.elapsed, **self._metrics(), **self.totals, **extra}

    # --------------------------------------------------------------- dynamics
    def _resolve_moves(self, commands):
        k = self.n_drones
        proposed = self.positions + MOVES[commands]
        proposed[self.lost] = self.positions[self.lost]
        invalid = np.any((proposed < 0) | (proposed >= self.config.grid_size), axis=1)
        proposed[invalid] = self.positions[invalid]
        blocked = np.zeros(k, dtype=bool)
        while True:
            reject = np.zeros(k, dtype=bool)
            for i in range(k):
                for j in range(i+1, k):
                    if self.lost[i] or self.lost[j]:
                        continue
                    same = np.array_equal(proposed[i], proposed[j])
                    at_base = np.array_equal(proposed[i], self.base)
                    swap = (np.array_equal(proposed[i], self.positions[j]) and
                            np.array_equal(proposed[j], self.positions[i]) and
                            not np.array_equal(self.positions[i], self.positions[j]))
                    if (same and not at_base) or swap:
                        for m in (i, j):
                            if not np.array_equal(proposed[m], self.positions[m]):
                                reject[m] = True
            if not reject.any():
                break
            blocked |= reject
            proposed[reject] = self.positions[reject]
        self.positions = proposed
        return int(blocked.sum())

    def _energy_and_charging(self, commands, previous):
        c = self.config
        at_base = np.all(self.positions == self.base, axis=1)
        was_at_base = np.all(previous == self.base, axis=1)
        ground_stay = at_base & was_at_base & (commands == 0) & ~self.lost
        eligible = ground_stay & (self.batteries < c.battery_capacity)
        owner = self.charger_owner
        if owner < 0 or not eligible[owner]:
            ids = np.flatnonzero(eligible)
            owner = min(ids, key=lambda i: (self.batteries[i], i)) if len(ids) else -1
        waiting = int(eligible.sum()) - int(owner >= 0)
        completions = 0
        if owner >= 0:
            self.batteries[owner] = min(c.battery_capacity,
                                        self.batteries[owner] + c.recharge_rate)
            completions = int(self.batteries[owner] == c.battery_capacity)
        self.charger_owner = int(owner) if not completions else -1
        costs = np.where(commands == 0, c.hover_cost, c.move_cost)
        costs[ground_stay | self.lost] = 0
        spent = np.minimum(self.batteries, costs)
        self.batteries[:] = np.maximum(self.batteries - costs, 0)
        newly_lost = ~self.lost & ~at_base & (self.batteries <= 0)
        self.lost |= newly_lost
        returns = int(np.count_nonzero(at_base & ~was_at_base & ~self.lost))
        return float(spent.sum()), int(newly_lost.sum()), returns, completions, waiting

    def _footprints(self):
        """Per-drone boolean sensor footprints, shape (n_drones, n, n)."""
        n, radius = self.config.grid_size, self.config.sensor_radius
        out = np.zeros((self.n_drones, n, n), dtype=bool)
        for i, ((r, col), lost) in enumerate(zip(self.positions, self.lost, strict=True)):
            if not lost:
                out[i, max(0, r-radius):min(n, r+radius+1),
                    max(0, col-radius):min(n, col+radius+1)] = True
        return out

    def _evolve_fire(self):
        c, n = self.config, self.config.grid_size
        draws = self.np_random.random((n, n))
        change_draw = self.np_random.random()
        new_wind = int(self.np_random.integers(0, 5))
        survival = np.ones((n, n), dtype=np.float64)
        for dr, dc in MOVES[1:]:
            neighbours = np.zeros((n, n), dtype=bool)
            r0, r1 = max(0, dr), n+min(0, dr)
            c0, c1 = max(0, dc), n+min(0, dc)
            neighbours[r0:r1, c0:c1] = self.burning[r0-dr:r1-dr, c0-dc:c1-dc]
            boost = c.wind_multiplier if np.array_equal([dr, dc], WIND[self.wind]) else 1
            probability = np.clip(c.spread_probability*boost*self.terrain, 0, 1)
            survival *= np.where(neighbours, 1-probability, 1)
        ignite = ~self.burning & (self.fuel > 0) & (draws < 1-survival)
        self.fuel[self.burning] = np.maximum(self.fuel[self.burning]-1, 0)
        self.burning = (self.burning & (self.fuel > 0)) | ignite
        if change_draw < c.wind_change_probability:
            self.wind = new_wind

    def step(self, commands):
        if self._done:
            raise RuntimeError('call reset() before stepping a finished environment')
        commands = np.asarray(commands)
        if (commands.shape != (self.n_drones,) or
                not np.issubdtype(commands.dtype, np.integer) or
                np.any((commands < 0) | (commands > 4))):
            raise ValueError(f'expected {self.n_drones} integer commands in [0, 4]')
        commands = commands.astype(np.int64)
        previous = self.positions.copy()
        collisions = self._resolve_moves(commands)
        energy, losses, returns, completions, waiting = self._energy_and_charging(
            commands, previous)
        self.ages[:] = np.minimum(self.ages+1, self.age_limit)
        weights = self.priority_weights()
        footprints = self._footprints()
        mask = footprints.any(axis=0)
        duplicates = int((footprints.sum(axis=0) > 1).sum())
        gain = float((weights[mask]*np.minimum(
            self.ages[mask]/self.config.inspection_deadline, 1)).sum()/weights.sum())
        self.ages[mask] = 0
        inspection_metrics = self._metrics(weights)
        c = self.config
        components = {
            'freshness_gain': c.freshness_reward*gain,
            'overdue': -c.overdue_penalty*inspection_metrics['overdue_fraction'],
            'energy': -c.energy_penalty*energy/c.battery_capacity,
            'collisions': -c.collision_penalty*collisions,
            'losses': -c.loss_penalty*losses,
            'mission_failure': (-c.overdue_penalty*(c.horizon-self.elapsed-1)
                                if self.lost.all() else 0.0),
        }
        reward = float(sum(components.values()))
        self._evolve_fire()
        self.elapsed += 1
        self._done = bool(self.lost.all() or self.elapsed >= c.horizon)
        for key, value in dict(energy_used=energy, collision_attempts=collisions,
                               safe_returns=returns, charge_completions=completions,
                               inspected_cells=int(mask.sum()), episode_return=reward,
                               duplicate_cells=duplicates,
                               charger_wait_steps=waiting).items():
            self.totals[key] += value
        info = self._info(**inspection_metrics, reward_components=components,
                          new_losses=losses, inspected_cells=int(mask.sum()),
                          safe_returns=returns, charge_completions=completions,
                          collision_attempts=collisions, energy_used=energy,
                          duplicate_cells=duplicates, charger_wait_steps=waiting)
        return reward, self._done, info

    @property
    def done(self):
        return self._done
