"""Nonintervening command, return-energy and replay diagnostics."""

import numpy as np

from wild_drone.environment import MOVES, decode_action
from wild_drone.experiments import run_episode

TOLERANCE = 1e-4  # Battery units; accommodate normalized float32 replay observations.


def return_status(positions, batteries, lost, base, config):
    distances = np.abs(positions-base).sum(axis=-1)
    margin = batteries-distances*config.move_cost
    away = ~lost & (distances > 0)
    return {'distances': distances, 'margin': margin,
            'unreachable': away & (margin < -TOLERANCE),
            'critical': away & (margin >= -TOLERANCE) &
            (margin <= 3*config.move_cost+TOLERANCE)}


def exploration_summary(settings, steps):
    """Expected branch selections, not observed random-action counts."""
    start, end, decay = (settings[key] for key in (
        'epsilon_start', 'epsilon_end', 'epsilon_decay_steps'))
    ramp = min(steps, decay)
    expected = ramp*start+(end-start)*ramp*(ramp-1)/(2*decay)+(steps-ramp)*end
    return {'steps': steps, 'expected_random_selections': expected,
            'expected_random_fraction': expected/steps if steps else None,
            'last_action_epsilon': start+min(1, (steps-1)/decay)*(end-start) if steps else None,
            'steps_at_epsilon_floor': max(0, steps-decay)}


class _Recorder:
    def __init__(self, policy, trace):
        self.policy, self.keep_trace = policy, trace
        self.pending, self.env = None, None
        self.trace, self.actions = [], []
        self.first_unreachable = {}
        self.stats = dict(live_drone_steps=0, wall_commands=0, unmoved_move_commands=0,
                          unsafe_drone_steps=0, critical_drone_steps=0,
                          critical_return_commands=0, critical_return_moves=0)

    def complete(self):
        if self.pending is None:
            return
        before, commands, wall, status = self.pending
        env = self.env
        live = ~before['lost']
        unchanged = np.all(env.positions == before['positions'], axis=1)
        distance = np.abs(env.positions-env.base).sum(axis=1)
        margin = env.batteries-distance*env.config.move_cost
        self.stats['unmoved_move_commands'] += int((live & (commands != 0) & unchanged).sum())
        self.stats['critical_return_moves'] += int((status['critical'] &
                                                   (distance < status['distances'])).sum())
        for drone in range(3):
            # Include a loss transition that crosses the bound immediately upon departure.
            deficit = live[drone] and distance[drone] > 0 and margin[drone] < -TOLERANCE
            if deficit:
                self.first_unreachable.setdefault(drone, env.elapsed)
            if self.keep_trace:
                self.trace.append({
                    'step': env.elapsed, 'drone': drone, 'command': int(commands[drone]),
                    'row_before': int(before['positions'][drone, 0]),
                    'col_before': int(before['positions'][drone, 1]),
                    'battery_before': float(before['batteries'][drone]),
                    'distance_before': int(status['distances'][drone]),
                    'return_margin_before': float(status['margin'][drone]),
                    'live_before': bool(live[drone]),
                    'critical_before': bool(status['critical'][drone]),
                    'wall_command': bool(wall[drone]),
                    'row_after': int(env.positions[drone, 0]),
                    'col_after': int(env.positions[drone, 1]),
                    'battery_after': float(env.batteries[drone]),
                    'distance_after': int(distance[drone]),
                    'return_margin_after': float(margin[drone]),
                    'lost_after': bool(env.lost[drone]),
                    'new_loss': bool(live[drone] and env.lost[drone]),
                    'unmoved_move': bool(live[drone] and commands[drone] != 0 and unchanged[drone]),
                })
        self.pending = None

    def act(self, env, observation):
        self.complete()  # Observe the preceding transition without an extra env.step.
        self.env = env
        before = {'positions': env.positions.copy(), 'batteries': env.batteries.copy(),
                  'lost': env.lost.copy()}
        status = return_status(env.positions, env.batteries, env.lost, env.base, env.config)
        for drone in np.flatnonzero(status['unreachable']):
            self.first_unreachable.setdefault(int(drone), env.elapsed)
        action = self.policy.act(env, observation)
        commands = decode_action(action)
        proposed = env.positions+MOVES[commands]
        wall = ~env.lost & np.any((proposed < 0) | (proposed >= env.config.grid_size), axis=1)
        distance = np.abs(proposed-env.base).sum(axis=1)
        self.stats['live_drone_steps'] += int((~env.lost).sum())
        self.stats['wall_commands'] += int(wall.sum())
        self.stats['unsafe_drone_steps'] += int(status['unreachable'].sum())
        self.stats['critical_drone_steps'] += int(status['critical'].sum())
        self.stats['critical_return_commands'] += int((status['critical'] &
                                                      (distance < status['distances'])).sum())
        self.actions.append(action)
        self.pending = before, commands, wall, status
        return action


def behavior_episode(config, policy, scenario_seed, *, name, training_seed=None, trace=False):
    recorder = _Recorder(policy, trace)
    metrics = run_episode(config, recorder, scenario_seed, name=name, training_seed=training_seed)
    recorder.complete()  # Include the final transition even when the fleet is lost.
    counts = np.bincount(recorder.actions, minlength=125)
    longest, streak, previous = 0, 0, None
    for action in recorder.actions:
        streak = streak+1 if action == previous else 1
        longest, previous = max(longest, streak), action
    stats = recorder.stats
    row = metrics | stats | {
        'first_unreachable_step': min(recorder.first_unreachable.values(), default=None),
        'first_unreachable_drones': len(recorder.first_unreachable),
        'distinct_joint_actions': int(np.count_nonzero(counts)),
        'dominant_joint_fraction': float(counts.max()/len(recorder.actions)),
        'longest_joint_repeat': longest,
        'critical_return_move_fraction': stats['critical_return_moves']/
        stats['critical_drone_steps'] if stats['critical_drone_steps'] else None}
    return row, recorder.trace


def replay_summary(state):
    """Coverage of the final retained buffer, not all historical training transitions."""
    from wild_drone.config import EnvConfig
    config = EnvConfig(**state['env_config'])
    replay = state['replay']
    if replay is None:
        return {'transitions': 0}
    result = {'transitions': len(replay['actions'])}
    observations = {}
    for label, key in [('before', 'observations'), ('after', 'next_observations')]:
        data = replay[key]
        drones = data['drones'].numpy()
        positions = np.rint(drones[:, :, :2]*(config.grid_size-1)).astype(int)
        base = np.rint(data['globals'][:, 9:11].numpy()*(config.grid_size-1)).astype(int)
        batteries = drones[:, :, 2]*config.battery_capacity
        lost = drones[:, :, 3] > .5
        status = return_status(positions, batteries, lost, base[:, None], config)
        live = ~lost
        result[label+'_live_drone_states'] = int(live.sum())
        result[label+'_unsafe_drone_states'] = int(status['unreachable'].sum())
        result[label+'_live_at_base_states'] = int((live & (status['distances'] == 0)).sum())
        result[label+'_critical_drone_states'] = int(status['critical'].sum())
        observations[label] = (lost, status, batteries)
    before_lost, before_status, before_battery = observations['before']
    after_lost, after_status, after_battery = observations['after']
    result['terminal_transitions'] = int(replay['terminals'].sum())
    result['loss_transitions'] = int(((~before_lost & after_lost).any(axis=1)).sum())
    result['return_transitions'] = int(((~after_lost & (before_status['distances'] > 0) &
                                        (after_status['distances'] == 0)).any(axis=1)).sum())
    result['charging_transitions'] = int(((~after_lost & (after_battery > before_battery+
                                                         TOLERANCE)).any(axis=1)).sum())
    actions = replay['actions'].numpy()
    result['distinct_joint_actions'] = int(np.unique(actions).size)
    result['dominant_joint_fraction'] = float(
        np.bincount(actions, minlength=125).max()/len(actions))
    return result
