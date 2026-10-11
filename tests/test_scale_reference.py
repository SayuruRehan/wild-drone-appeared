import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

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


def test_audit_rejects_metrics_removed_from_all_summary_files(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / 'scripts'
    spec = importlib.util.spec_from_file_location('reference_audit',
                                                directory / 'audit_scale_reference.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    validate = getattr(module, 'validate_metric_schema', None)
    assert callable(validate), 'audit must validate an independent expected metric schema'
    metrics = {'return', 'weighted_freshness', 'overdue_fraction', 'energy_used',
               'total_losses', 'safe_returns', 'charge_completions', 'collision_attempts',
               'steps', 'fleet_failure', 'intact_fleet'}
    valid = dict.fromkeys(metrics | {'policy', 'training_seed', 'worlds'}, 0)
    validate(valid, paired=False)
    del valid['overdue_fraction']
    with pytest.raises(AssertionError, match='metric schema'):
        validate(valid, paired=False)
    paired = dict.fromkeys(metrics | {'comparison_policy', 'training_seed', 'scenario_seed'}, 0)
    validate(paired, paired=True)
    paired['unexpected_metric'] = 0
    with pytest.raises(AssertionError, match='metric schema'):
        validate(paired, paired=True)


def test_audit_rejects_changed_environment_or_removed_parent_identity():
    directory = Path(__file__).resolve().parents[1] / 'scripts'
    spec = importlib.util.spec_from_file_location('reference_input_audit',
                                                directory / 'audit_scale_reference.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    validate = getattr(module, 'validate_launch', None)
    assert callable(validate), 'audit must bind the whole configuration to the frozen parent'
    parent = directory.parent / 'docs/experiments/scale-pilot-2026-10-11/launch.json'
    launch = {'env_config': json.loads(parent.read_text())['env_config'],
              'parent_input_sha256': dict.fromkeys([
                  'docs/experiments/scale-pilot-2026-10-11/' + name
                  for name in ('launch.json', 'missions/episodes.csv', 'checkpoints.json',
                               'completion.json')], 'placeholder')}
    validate(launch)
    changed = copy.deepcopy(launch)
    changed['env_config']['battery_capacity'] += 1
    with pytest.raises(AssertionError, match='environment'):
        validate(changed)
    del launch['parent_input_sha256']['docs/experiments/scale-pilot-2026-10-11/launch.json']
    with pytest.raises(AssertionError, match='parent input'):
        validate(launch)


def test_independent_world_audit_rejects_action_or_packet_drift(monkeypatch):
    module = runner(monkeypatch)
    spec = importlib.util.spec_from_file_location('reference_world_audit',
                                                Path(module.__file__).with_name(
                                                    'audit_scale_reference.py'))
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    check = getattr(audit, 'audit_world', None)
    assert callable(check), 'audit must independently verify each complete world packet'
    config = EnvConfig(grid_size=8, horizon=4)
    episode, trace, vectors = module.rollout(config, module.ExactInspectionPolicy(config),
                                            0, name='exact_inspection')
    def strings(row):
        return {k: '' if v is None else str(v) for k, v in row.items()}

    episode, trace = strings(episode), [strings(r) for r in trace]
    verified, drift = check(config, episode, trace, vectors['scores'], vectors['masks'])
    assert verified['steps'] == 4 and drift < 1e-12
    bad = copy.deepcopy(trace)
    bad[0]['action'] = str((int(bad[0]['action']) + 1) % 125)
    with pytest.raises(AssertionError):
        check(config, episode, bad, vectors['scores'], vectors['masks'])
    with pytest.raises(AssertionError):
        check(config, episode, trace, vectors['scores'][:-1], vectors['masks'][:-1])
