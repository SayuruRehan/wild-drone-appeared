import copy

import numpy as np

from wild_drone.baselines import GreedyPolicy, RandomPolicy, RollingHorizonPolicy, SweepPolicy
from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv


def test_baselines_produce_joint_actions():
    env = WildfireFleetEnv(EnvConfig(grid_size=8, horizon=20))
    for policy in [RandomPolicy(7), GreedyPolicy(), SweepPolicy(),
                   RollingHorizonPolicy(seed=7, horizon=2, candidates=3, samples=2)]:
        obs, _ = env.reset(seed=7)
        for _ in range(5):
            action = policy.act(env, obs)
            assert env.action_space.contains(action)
            obs, _, done, _, _ = env.step(action)
            if done:
                break


def test_battery_aware_baselines_return_safely():
    for policy in [GreedyPolicy(), SweepPolicy()]:
        env = WildfireFleetEnv(EnvConfig(grid_size=8, horizon=100, battery_capacity=20))
        obs, _ = env.reset(seed=9)
        for _ in range(100):
            obs, _, done, _, _ = env.step(policy.act(env, obs))
            if done:
                break
        assert env.totals['safe_returns'] > 0
        assert not env.lost.any()
        assert env.totals['collision_attempts'] == 0


def test_planning_does_not_change_live_state_or_live_rng():
    env = WildfireFleetEnv(EnvConfig(grid_size=8, horizon=20))
    obs, _ = env.reset(seed=4)
    before = copy.deepcopy(env)
    policy = RollingHorizonPolicy(seed=42, horizon=3, candidates=4, samples=2)
    policy.act(env, obs)
    assert env.np_random.bit_generator.state == before.np_random.bit_generator.state
    for key, values in obs.items():
        np.testing.assert_array_equal(values, env._observation()[key])
    assert env.elapsed == before.elapsed


def test_planner_cannot_predict_live_rng_stream():
    env = WildfireFleetEnv(EnvConfig(grid_size=8, horizon=20))
    obs, _ = env.reset(seed=14)
    other = copy.deepcopy(env)
    other.np_random = np.random.default_rng(932)
    a = RollingHorizonPolicy(seed=4, horizon=3, candidates=4, samples=2).act(env, obs)
    b = RollingHorizonPolicy(seed=4, horizon=3, candidates=4, samples=2).act(other, obs)
    assert a == b
