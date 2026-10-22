import numpy as np
import pytest

from wild_drone.behavior import behavior_episode, exploration_summary, return_status
from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv, encode_action
from wild_drone.experiments import run_episode


class FixedPolicy:
    def __init__(self, action):
        self.action = action

    def act(self, env, observation):
        return self.action


def test_return_status_excludes_ground_and_lost_drones_and_detects_energy_deficit():
    c = EnvConfig(grid_size=8, horizon=8, move_cost=2)
    status = return_status(np.array([[4, 4], [4, 6], [4, 7]]), np.array([0., 3., 0.]),
                           np.array([False, False, True]), np.array([4, 4]), c)
    np.testing.assert_array_equal(status['distances'], [0, 2, 3])
    np.testing.assert_array_equal(status['unreachable'], [False, True, False])
    np.testing.assert_array_equal(status['critical'], [False, False, False])


def test_behavior_trace_preserves_mission_and_records_terminal_step():
    c = EnvConfig(grid_size=8, horizon=8, battery_capacity=2)
    action = encode_action([1, 0, 0])
    expected = run_episode(c, FixedPolicy(action), 5000, name='fixed')
    actual, trace = behavior_episode(c, FixedPolicy(action), 5000, name='fixed', trace=True)
    for key in expected:
        if key != 'decision_ms':
            assert actual[key] == expected[key]
    assert len(trace) == 3*actual['steps']
    losses = [row for row in trace if row['new_loss']]
    assert len(losses) == 1 and losses[0]['step'] == 2
    assert actual['first_unreachable_step'] == 2
    assert actual['first_unreachable_drones'] == 1


def test_wall_commands_are_separate_from_collision_blocks():
    c = EnvConfig(grid_size=4, horizon=6, battery_capacity=80)
    row, trace = behavior_episode(c, FixedPolicy(encode_action([1, 0, 0])), 5000,
                                  name='fixed', trace=True)
    assert row['wall_commands'] == 4
    assert row['unmoved_move_commands'] == 4
    assert row['collision_attempts'] == 0
    assert sum(r['wall_command'] for r in trace) == 4
    assert row['longest_joint_repeat'] == 6 and row['dominant_joint_fraction'] == 1


def test_recording_does_not_consume_live_environment_randomness():
    from wild_drone.baselines import RandomPolicy
    c = EnvConfig(grid_size=8, horizon=12)
    expected = run_episode(c, RandomPolicy(10), 5000, name='random')
    actual, _ = behavior_episode(c, RandomPolicy(10), 5000, name='random')
    assert {k: v for k, v in actual.items() if k in expected and k != 'decision_ms'} == (
        {k: v for k, v in expected.items() if k != 'decision_ms'})


def test_exploration_summary_uses_pre_action_step_schedule():
    settings = {'epsilon_start': 1., 'epsilon_end': .2, 'epsilon_decay_steps': 4}
    # Five actual selection probabilities: 1, .8, .6, .4, .2.
    report = exploration_summary(settings, 5)
    assert report['expected_random_selections'] == pytest.approx(3.)
    assert report['last_action_epsilon'] == pytest.approx(.2)
    assert report['steps_at_epsilon_floor'] == 1


def test_replay_summary_detects_real_return_and_charge_transitions():
    from wild_drone.behavior import replay_summary
    from wild_drone.training_io import pack_replay
    c = EnvConfig(grid_size=8, horizon=8, battery_capacity=2, recharge_rate=1)
    env = WildfireFleetEnv(c)
    env.reset(seed=5000)
    env.positions[0] = env.base+[0, 1]
    replay = []
    for action in [encode_action([3, 0, 0]), 0]:
        obs = env._observation()
        next_obs, reward, terminal, _, _ = env.step(action)
        replay.append((obs, action, reward, next_obs, terminal))
    result = replay_summary({'env_config': c.to_dict(), 'replay': pack_replay(replay)})
    assert result['return_transitions'] == 1 and result['charging_transitions'] == 1
    assert result['transitions'] == 2 and result['distinct_joint_actions'] == 2
    assert result['before_unsafe_drone_states'] == 0


def test_critical_return_command_blocked_by_collision_is_not_a_return_move():
    from wild_drone.behavior import _Recorder
    c = EnvConfig(grid_size=8, horizon=8)
    env = WildfireFleetEnv(c)
    env.reset(seed=5000)
    env.positions[:] = [[4, 6], [4, 5], [0, 0]]
    env.batteries[0] = 3  # Distance two, margin one: critical.
    recorder = _Recorder(FixedPolicy(encode_action([3, 0, 0])), False)
    action = recorder.act(env, env._observation())
    env.step(action)
    recorder.complete()
    assert recorder.stats['critical_return_commands'] == 1
    assert recorder.stats['critical_return_moves'] == 0
    assert recorder.stats['unmoved_move_commands'] == 1
