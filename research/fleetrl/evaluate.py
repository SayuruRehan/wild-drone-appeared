"""Fly one mission and report the study metrics (full-horizon accounting)."""

import time

import numpy as np

from fleetrl.env import FleetEnv


def run_episode(config, n_drones, policy, seed, record=False):
    env = FleetEnv(config, n_drones)
    env.reset(seed)
    if hasattr(policy, 'reset'):
        policy.reset()
    freshness, overdue, decision_ms, trace = [], [], [], []
    done = False
    while not done:
        t0 = time.perf_counter()
        commands = policy.act_commands(env)
        decision_ms.append(1000*(time.perf_counter()-t0))
        _, done, info = env.step(commands)
        freshness.append(info['weighted_freshness'])
        overdue.append(info['overdue_fraction'])
        if record:
            trace.append(np.asarray(commands).tolist())
    # Full-horizon accounting: if the fleet is lost early, missing steps count as zero
    # freshness and fully overdue (same convention as the frozen study).
    missing = config.horizon - len(freshness)
    freshness += [0.0]*missing
    overdue += [1.0]*missing
    totals = env.totals
    out = {
        'seed': seed,
        'n_drones': n_drones,
        'weighted_freshness': float(np.mean(freshness)),
        'overdue_fraction': float(np.mean(overdue)),
        'return': float(totals['episode_return']),
        'losses': int(env.lost.sum()),
        'energy_used': float(totals['energy_used']),
        'duplicate_cells_per_step': totals['duplicate_cells']/max(1, env.elapsed),
        'charger_wait_steps': int(totals['charger_wait_steps']),
        'collision_attempts': int(totals['collision_attempts']),
        'charge_completions': int(totals['charge_completions']),
        'decision_ms': float(np.mean(decision_ms)),
    }
    if record:
        out['trace'] = trace
    return out
