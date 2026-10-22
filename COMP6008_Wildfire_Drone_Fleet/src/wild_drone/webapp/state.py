"""Serialize actual simulator states; never approximate physics in the browser."""

import hashlib
import json

import numpy as np

from wild_drone.environment import decode_action


def observation_hash(observation):
    text = json.dumps({k: v.tolist() for k, v in observation.items()},
                      sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def snapshot(env, info=None, *, action=None, previous=None, decision=None):
    metrics = env._metrics() if info is None else {
        k: info[k] for k in ('weighted_freshness', 'overdue_fraction', 'total_losses')}
    events = []
    if previous is not None:
        commands = decode_action(action)
        for i in range(3):
            prefix = f'Drone {i+1}'
            if env.lost[i] and not previous['lost'][i]:
                events.append(f'{prefix} lost: battery exhausted away from base.')
            elif env.batteries[i] > previous['batteries'][i]:
                events.append(f'{prefix} charging at the shared base.')
            elif (np.array_equal(env.positions[i], env.base) and
                  not np.array_equal(previous['positions'][i], env.base)):
                events.append(f'{prefix} returned to base; charging requires a stay command.')
            elif (commands[i] != 0 and not previous['lost'][i] and
                  np.array_equal(env.positions[i], previous['positions'][i])):
                events.append(f'{prefix} movement blocked by a boundary or another drone.')
        if info['charge_completions']:
            events.append('A recharge completed; the charger is available again.')
        if env.lost.all():
            events.append('Fleet lost. The remaining mission is counted as unserved.')
    return {
        'step': env.elapsed, 'burning': env.burning.astype(int).ravel().tolist(),
        'fuel': env.fuel.ravel().tolist(), 'ages': env.ages.ravel().tolist(),
        'positions': env.positions.tolist(), 'batteries': env.batteries.tolist(),
        'lost': env.lost.tolist(), 'wind': env.wind, 'charger_owner': env.charger_owner,
        'commands': decode_action(action).tolist() if action is not None else [0, 0, 0],
        'action': action, 'metrics': metrics, 'reward': sum(
            info['reward_components'].values()) if info else 0,
        'reward_components': info['reward_components'] if info else {},
        'cumulative_return': env.totals['episode_return'],
        'energy_used': env.totals['energy_used'], 'safe_returns': env.totals['safe_returns'],
        'charge_completions': env.totals['charge_completions'],
        'collision_attempts': env.totals['collision_attempts'],
        'events': events, 'decision': decision, 'terminal': bool(env._done),
    }


def initial_packet(env, *, policy, world, training_seed=None, kind='sandbox'):
    return {
        'format_version': 1, 'kind': kind, 'policy': policy, 'world': world,
        'training_seed': training_seed, 'config': env.config.to_dict(),
        'base': env.base.tolist(), 'terrain': env.terrain.ravel().tolist(),
        'priority': env.priority.ravel().tolist(),
    }


def mission_summary(env, frames):
    rows = frames[1:]
    horizon = env.config.horizon
    return {
        'return': env.totals['episode_return'], 'steps': env.elapsed,
        'weighted_freshness': sum(f['metrics']['weighted_freshness'] for f in rows)/horizon,
        'overdue_fraction': (sum(f['metrics']['overdue_fraction'] for f in rows) +
                             horizon-env.elapsed)/horizon,
        'total_losses': int(env.lost.sum()), 'fleet_failure': int(env.lost.all()),
        'intact_fleet': int(not env.lost.any()),
        **{k: env.totals[k] for k in ('energy_used', 'safe_returns', 'charge_completions',
                                     'collision_attempts')},
    }
