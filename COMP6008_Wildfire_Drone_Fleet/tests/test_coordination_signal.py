import csv
from dataclasses import replace

import numpy as np
import pytest
import torch

from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.learning import TrainConfig, train
from wild_drone.masked_learning import load_masked_policy, train_masked


def potential(env, **kwargs):
    from wild_drone.coordination import coordination_potential
    return coordination_potential(env, **kwargs)


def test_potential_rewards_freshness_and_approach_without_duplicate_drone_credit():
    env = WildfireFleetEnv(EnvConfig(grid_size=8, sensor_radius=0))
    env.reset(seed=0)
    env.burning[:] = False
    env.priority[:] = 1
    env.ages[:] = 0
    assert potential(env) == pytest.approx(1)
    env.ages[:] = env.config.inspection_deadline
    env.lost[:] = True
    assert potential(env) == 0
    env.lost[0] = False
    env.positions[0] = [0, 0]
    corner = potential(env)
    env.positions[0] = [3, 3]
    centre = potential(env)
    assert 0 < corner < centre < 1
    env.positions[:] = [3, 3]
    env.lost[:] = False
    assert potential(env) == centre
    env.positions[1:] = [[0, 0], [7, 7]]
    assert potential(env) > centre
    assert potential(env, terminal=True) == 0


@pytest.mark.parametrize('gamma', [0, .99, 1])
@pytest.mark.parametrize('early_loss', [False, True])
def test_discounted_shaping_changes_return_only_by_initial_state_constant(gamma, early_loss):
    env = WildfireFleetEnv(EnvConfig(grid_size=8, horizon=7, battery_capacity=.1
                                   if early_loss else 160))
    env.reset(seed=0)
    if early_loss:
        env.positions[:] = [[1, 1], [1, 3], [1, 6]]
    initial = before = potential(env)
    shaped_total = original_total = 0
    step = 0
    done = False
    while not done:
        _, reward, terminal, truncated, _ = env.step(31)  # all move north
        done = terminal or truncated
        after = potential(env, terminal=done)
        shaped_total += gamma**step*(reward+10*(gamma*after-before))
        original_total += gamma**step*reward
        before = after
        step += 1
    assert step == (1 if early_loss else 7)
    assert shaped_total-original_total == pytest.approx(-10*initial, abs=1e-10)


@pytest.mark.parametrize('algorithm', ['dqn', 'a2c', 'reinforce'])
def test_training_signal_is_separate_from_mission_metrics_and_checkpointed(algorithm, tmp_path):
    from wild_drone.coordination import SIGNAL_RULE
    from wild_drone.study_cache import validate_cached_fit

    config = EnvConfig(grid_size=8, horizon=8)
    settings = TrainConfig(episodes=1, batch_size=4, warmup_steps=4, n_steps=4,
                           progress_interval=0, coordination_shaping_weight=10)
    train_masked(algorithm, config, settings, tmp_path)
    with (tmp_path/'training.csv').open() as handle:
        row = next(csv.DictReader(handle))
    assert float(row['return']) == float(row['episode_return'])
    assert float(row['learning_return']) == pytest.approx(
        float(row['return'])+float(row['shaping_return']))
    assert float(row['discounted_shaping_return']) == pytest.approx(
        -10*float(row['initial_potential']), abs=1e-10)
    assert float(row['shaping_return']) != 0
    policy = load_masked_policy(tmp_path/'policy.pt')
    assert policy.train_config == settings
    assert validate_cached_fit(True, algorithm, config, settings, tmp_path)
    with pytest.raises(ValueError, match='train_config'):
        validate_cached_fit(True, algorithm, config, replace(settings,
                            coordination_shaping_weight=0), tmp_path)
    data = torch.load(tmp_path/'policy.pt', weights_only=True)
    assert data['learning_signal'] == SIGNAL_RULE
    del data['learning_signal']
    torch.save(data, tmp_path/'policy.pt')
    with pytest.raises(ValueError, match='learning signal'):
        load_masked_policy(tmp_path/'policy.pt')


def test_unmasked_training_rejects_unsupported_signal_before_writing(tmp_path):
    settings = TrainConfig(coordination_shaping_weight=10)
    with pytest.raises(ValueError, match='masked'):
        train('dqn', EnvConfig(grid_size=8), settings, tmp_path/'fit')
    assert not (tmp_path/'fit').exists()


@pytest.mark.parametrize('weight', [-1, np.nan, np.inf, True])
def test_invalid_shaping_weight_is_rejected(weight):
    with pytest.raises(ValueError, match='coordination_shaping_weight'):
        TrainConfig(coordination_shaping_weight=weight)


@pytest.mark.parametrize('algorithm', ['dqn', 'a2c', 'reinforce'])
def test_update_targets_receive_shaped_rewards(algorithm, tmp_path, monkeypatch):
    import wild_drone.masked_learning as learning

    config = EnvConfig(grid_size=8, horizon=4)
    expected, received = [], []
    original_step = WildfireFleetEnv.step

    def observed_step(env, action):
        before = potential(env)
        result = original_step(env, action)
        _, reward, terminal, truncated, _ = result
        expected.append(reward+10*(.99*potential(env, terminal=terminal or truncated)-before))
        return result

    monkeypatch.setattr(WildfireFleetEnv, 'step', observed_step)
    target_name = {'dqn': 'td_targets', 'a2c': 'discounted_returns',
                   'reinforce': 'monte_carlo_terms'}[algorithm]
    original_target = getattr(learning, target_name)

    def observed_target(*args, **kwargs):
        rewards = args[1] if algorithm == 'reinforce' else args[0]
        received.extend(rewards.tolist() if isinstance(rewards, torch.Tensor) else rewards)
        return original_target(*args, **kwargs)

    monkeypatch.setattr(learning, target_name, observed_target)
    settings = TrainConfig(episodes=1, batch_size=1, replay_size=1, warmup_steps=1,
                           n_steps=4, progress_interval=0, coordination_shaping_weight=10)
    train_masked(algorithm, config, settings, tmp_path)
    np.testing.assert_allclose(received, expected, rtol=1e-6, atol=1e-7)


def test_zero_weight_skips_potential_and_legacy_cache_stays_valid(tmp_path, monkeypatch):
    import json

    import wild_drone.masked_learning as learning
    from wild_drone.study_cache import validate_cached_fit

    def forbidden(*args, **kwargs):
        raise AssertionError('zero weight must not compute the potential')

    monkeypatch.setattr(learning, 'coordination_potential', forbidden)
    config = EnvConfig(grid_size=8, horizon=4)
    settings = TrainConfig(episodes=1, progress_interval=0)
    train_masked('dqn', config, settings, tmp_path)
    with (tmp_path/'training.csv').open() as handle:
        row = next(csv.DictReader(handle))
    assert row['learning_return'] == row['return']
    assert float(row['shaping_return']) == 0
    summary_path = tmp_path/'training_summary.json'
    summary = json.loads(summary_path.read_text())
    summary['train_config'].pop('coordination_shaping_weight')
    summary_path.write_text(json.dumps(summary))
    data = torch.load(tmp_path/'policy.pt', weights_only=True)
    data['train_config'].pop('coordination_shaping_weight')
    torch.save(data, tmp_path/'policy.pt')
    assert validate_cached_fit(True, 'dqn', config, settings, tmp_path)


def probe_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1]/'scripts'/'run_coordination_probe.py'
    spec = importlib.util.spec_from_file_location('probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def screen_rows(delta_return=.6, delta_freshness=.02):
    return [dict(policy=policy, scenario_seed=seed, training_seed=0 if 'double' in policy else None,
                 return_=delta_return if policy == 'shaped double dqn' else 0,
                 weighted_freshness=.2+delta_freshness if policy == 'shaped double dqn' else .2,
                 total_losses=0) for policy in ['shaped double dqn', 'unshaped double dqn',
                                               'masked random', 'greedy']
            for seed in range(5700, 5720)]


def test_gate_requires_both_original_metrics_and_zero_candidate_losses():
    probe = probe_script()
    rows = screen_rows()
    for row in rows:
        row['return'] = row.pop('return_')
    gate, pairs = probe.screen_gate(rows)
    assert gate['passed'] and len(pairs) == 40
    for name, value in [('return', .4), ('weighted_freshness', .205), ('total_losses', 1)]:
        damaged = [dict(row) for row in rows]
        for row in damaged:
            if row['policy'] == 'shaped double dqn':
                row[name] = value
        assert not probe.screen_gate(damaged)[0]['passed']
    with pytest.raises(ValueError, match='complete'):
        probe.screen_gate(rows[:-1])
    with pytest.raises(ValueError, match='complete'):
        probe.screen_gate(rows+[rows[0]])
