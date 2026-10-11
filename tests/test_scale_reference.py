import copy
import importlib.util
from pathlib import Path

import numpy as np

from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.masked_learning import action_mask


def runner(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / 'scripts'
    path = directory / 'run_scale_reference.py'
    assert path.is_file(), 'controlled reference runner is missing'
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('scale_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_uses_actual_inspection_rewards_and_preserves_state(monkeypatch):
    module = runner(monkeypatch)
    config = EnvConfig(grid_size=8, horizon=20, battery_capacity=12)
    env = WildfireFleetEnv(config)
    observation, _ = env.reset(seed=3)
    # Include a collision, a depleted drone, and out-of-grid proposals in the
    # all-action packet; a geometric footprint alone would score these wrongly.
    env.positions[:] = [[0, 0], [0, 1], [1, 0]]
    env.batteries[:] = [1, 7, 9]
    before = copy.deepcopy(env)
    policy = module.ExactInspectionPolicy(config)
    selected = policy.act(env, observation)
    expected = []
    for action in range(125):
        clone = copy.deepcopy(before)
        _, _, _, _, info = clone.step(action)
        expected.append(info['reward_components']['freshness_gain'])
    np.testing.assert_allclose(policy.last_scores, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(policy.last_mask, action_mask(before))
    assert selected == int(np.argmax(np.where(policy.last_mask, expected, -np.inf)))
    for key in ('positions', 'batteries', 'ages', 'fuel', 'burning', 'lost'):
        np.testing.assert_array_equal(getattr(env, key), getattr(before, key))
    assert env.np_random.bit_generator.state == before.np_random.bit_generator.state
    assert env.totals == before.totals and env.elapsed == before.elapsed


def test_reference_first_index_ties_and_rollout_packet(monkeypatch):
    module = runner(monkeypatch)
    config = EnvConfig(grid_size=8, horizon=4, freshness_reward=0)
    env = WildfireFleetEnv(config)
    observation, _ = env.reset(seed=0)
    policy = module.ExactInspectionPolicy(config)
    assert policy.act(env, observation) == int(np.flatnonzero(action_mask(env))[0])
    episode, trace, vectors = module.rollout(config, policy, 0, name='exact_inspection')
    assert episode['steps'] == 4
    assert vectors['scores'].shape == (4, 125)
    assert vectors['masks'].dtype == bool
    assert np.where(vectors['masks'], vectors['scores'], -np.inf).argmax(1).tolist() == \
        [row['action'] for row in trace]
