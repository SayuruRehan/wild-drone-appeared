import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.learning import (
    FleetNetwork,
    TrainConfig,
    load_policy,
    resolve_device,
    tensor_observations,
    train,
)


def test_spatial_channels_preserve_maps_and_mark_live_drones_and_base():
    env = WildfireFleetEnv(EnvConfig(grid_size=8, horizon=10))
    observation, _ = env.reset(seed=7)
    env.positions[:] = [[1, 2], [3, 4], [5, 6]]
    env.lost[2] = True
    observation = env._observation()
    maps, vector = tensor_observations(observation, spatial_encoding=True)
    assert maps.shape == (1, 9, 8, 8) and vector.shape == (1, 24)
    np.testing.assert_array_equal(maps[0, :5].numpy(), observation['maps'])
    assert maps[0, 5, 1, 2] == 1 and maps[0, 6, 3, 4] == 1
    assert maps[0, 7].sum() == 0
    assert maps[0, 8, 4, 4] == 1


def test_reinforce_baseline_is_monte_carlo_and_saved_in_checkpoint(tmp_path):
    cfg = EnvConfig(grid_size=8, horizon=8)
    settings = TrainConfig(episodes=2, seed=4, monte_carlo_baseline=True)
    train('reinforce', cfg, settings, tmp_path)
    policy = load_policy(tmp_path/'policy.pt')
    assert policy.model.critic is not None
    assert policy.train_config.monte_carlo_baseline
    rows = list(csv.DictReader((tmp_path/'training.csv').open()))
    assert all(float(r['value_loss']) >= 0 for r in rows)


@pytest.mark.parametrize('algorithm', ['dqn', 'reinforce', 'a2c'])
@pytest.mark.parametrize('device', ['cpu', pytest.param(
    'cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA'))])
def test_resume_matches_uninterrupted_training_at_episode_boundary(algorithm, device, tmp_path):
    cfg = EnvConfig(grid_size=8, horizon=8)
    full = TrainConfig(episodes=4, seed=2, device=device, batch_size=4, warmup_steps=4,
                       n_steps=3, target_update_steps=5, checkpoint_interval=1,
                       progress_interval=0)
    a = train(algorithm, cfg, full, tmp_path/'full')
    partial = replace(full, episodes=2)
    train(algorithm, cfg, partial, tmp_path/'split')
    b = train(algorithm, cfg, full, tmp_path/'split',
              resume_from=tmp_path/'split'/'training_state.pt')
    assert a['episode_returns'] == b['episode_returns']
    assert a['updates'] == b['updates'] and a['last_loss'] == b['last_loss']
    ma = load_policy(tmp_path/'full'/'policy.pt').model.state_dict()
    mb = load_policy(tmp_path/'split'/'policy.pt').model.state_dict()
    for key in ma:
        torch.testing.assert_close(ma[key], mb[key], rtol=0, atol=0)
    assert len(list(csv.DictReader((tmp_path/'split'/'training.csv').open()))) == 4


def test_resume_rejects_changed_training_conditions_before_writing(tmp_path):
    cfg = EnvConfig(grid_size=8, horizon=6)
    settings = TrainConfig(episodes=1, checkpoint_interval=1, progress_interval=0)
    train('a2c', cfg, settings, tmp_path/'old')
    with pytest.raises(ValueError, match='resume'):
        train('a2c', cfg, replace(settings, episodes=2, gamma=0.5), tmp_path/'new',
              resume_from=tmp_path/'old'/'training_state.pt')
    assert not (tmp_path/'new').exists()


def test_validation_records_each_seed_and_best_checkpoint(tmp_path):
    settings = TrainConfig(episodes=3, seed=1, validation_seeds=(5000, 5001),
                           validation_interval=1, checkpoint_interval=1, progress_interval=0)
    train('a2c', EnvConfig(grid_size=8, horizon=6), settings, tmp_path)
    rows = list(csv.DictReader((tmp_path/'validation.csv').open()))
    assert len(rows) == 6
    assert {int(r['scenario_seed']) for r in rows} == {5000, 5001}
    assert (tmp_path/'best_policy.pt').exists()
    state = torch.load(tmp_path/'training_state.pt', weights_only=True)
    assert state['completed_episodes'] == 3
    assert len(state['validation_rows']) == 6
    assert state['validation_rows'][-1]['episode'] == 3
    assert json.loads((tmp_path/'training_summary.json').read_text())['validation_seeds'] == [
        5000, 5001]


def test_training_validation_overlap_is_rejected(tmp_path):
    settings = TrainConfig(episodes=3, validation_seeds=(2,), validation_interval=1)
    with pytest.raises(ValueError, match='overlap'):
        train('dqn', EnvConfig(grid_size=8, horizon=6), settings, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_legacy_checkpoint_remains_loadable(tmp_path):
    config = EnvConfig(grid_size=8, horizon=6)
    legacy = {k: v for k, v in TrainConfig().to_dict().items()
              if k not in {'device', 'checkpoint_interval', 'progress_interval',
                           'validation_interval', 'validation_seeds',
                           'spatial_encoding', 'monte_carlo_baseline'}}
    model = FleetNetwork(8, critic=False, spatial_encoding=False)
    torch.save({'algorithm': 'reinforce', 'env_config': config.to_dict(),
                'train_config': legacy, 'state_dict': model.state_dict()}, tmp_path/'legacy.pt')
    policy = load_policy(tmp_path/'legacy.pt')
    env = WildfireFleetEnv(config)
    obs, _ = env.reset(seed=5000)
    assert env.action_space.contains(policy.act(env, obs))
    assert not policy.train_config.spatial_encoding and policy.model.critic is None


def test_device_selection_fails_clearly_without_cuda():
    assert resolve_device('cpu').type == 'cpu'
    assert resolve_device('auto').type == ('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match='CUDA'):
            resolve_device('cuda')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires a CUDA runtime')
def test_cuda_training_update_and_portable_inference(tmp_path):
    settings = TrainConfig(episodes=2, device='cuda', batch_size=4, warmup_steps=4,
                           n_steps=3, checkpoint_interval=1, progress_interval=0)
    cfg = EnvConfig(grid_size=8, horizon=8)
    for algorithm in ['dqn', 'reinforce', 'a2c']:
        report = train(algorithm, cfg, settings, tmp_path/algorithm)
        assert report['updates'] > 0 and report['device'] == 'cuda'
        policy = load_policy(tmp_path/algorithm/'policy.pt', device='cpu')
        env = WildfireFleetEnv(cfg)
        observation, _ = env.reset(seed=5000)
        assert env.action_space.contains(policy.act(env, observation))


def test_extended_resume_keeps_best_model_and_updates_selection_seed_budget(tmp_path, monkeypatch):
    import wild_drone.learning as learning
    original = learning.run_episode
    calls = 0

    def decreasing_validation(*args, **kwargs):
        nonlocal calls
        row = original(*args, **kwargs)
        calls += 1
        return row | {'return': -float(calls)}

    monkeypatch.setattr(learning, 'run_episode', decreasing_validation)
    cfg = EnvConfig(grid_size=8, horizon=6)
    settings = TrainConfig(episodes=2, validation_seeds=(5000,), validation_interval=1,
                           checkpoint_interval=1, progress_interval=0)
    train('a2c', cfg, settings, tmp_path)
    train('a2c', cfg, replace(settings, episodes=4), tmp_path,
          resume_from=tmp_path/'training_state.pt')
    best = torch.load(tmp_path/'best_policy.pt', weights_only=True)
    assert best['completed_episodes'] == 1 and best['train_config']['episodes'] == 4
    assert best['selection']['validation_return'] == -1
    state = torch.load(tmp_path/'training_state.pt', weights_only=True)
    assert [s['resumed_episode'] for s in state['source_history']] == [0, 2]
    assert len(state['validation_rows']) == 4


def test_existing_policy_is_not_overwritten_without_resume(tmp_path):
    cfg, settings = EnvConfig(grid_size=8, horizon=6), TrainConfig(episodes=1)
    train('dqn', cfg, settings, tmp_path)
    before = (tmp_path/'policy.pt').read_bytes()
    with pytest.raises(ValueError, match='resume'):
        train('dqn', cfg, settings, tmp_path)
    assert (tmp_path/'policy.pt').read_bytes() == before


@pytest.mark.parametrize('interrupted_file', [
    'training_state.pt', 'policy.pt', 'best_policy.pt', 'training.csv',
    'validation.csv', 'training_summary.json'])
def test_first_checkpoint_interruption_can_restart_or_resume_exactly(
        interrupted_file, tmp_path, monkeypatch):
    import wild_drone.learning as learning
    cfg = EnvConfig(grid_size=8, horizon=6)
    settings = TrainConfig(episodes=2, seed=3, batch_size=4, warmup_steps=4,
                           checkpoint_interval=1, validation_interval=1,
                           validation_seeds=(5000,), progress_interval=0)
    expected = train('dqn', cfg, settings, tmp_path/'full')
    fired = False

    def fault_inject(original):
        def save(data, path):
            nonlocal fired
            if not fired and path.name == interrupted_file:
                fired = True
                raise InterruptedError('simulated disconnect')
            original(data, path)
        return save

    for name in ['atomic_torch_save', 'atomic_csv_save', 'atomic_json_save']:
        monkeypatch.setattr(learning, name, fault_inject(getattr(learning, name)))
    out = tmp_path/'interrupted'
    with pytest.raises(InterruptedError):
        train('dqn', cfg, settings, out)
    state = out/'training_state.pt'
    actual = train('dqn', cfg, settings, out, resume_from=state if state.exists() else None)
    assert actual['episode_returns'] == expected['episode_returns']
    assert actual['updates'] == expected['updates']
    a = load_policy(tmp_path/'full'/'policy.pt').model.state_dict()
    b = load_policy(out/'policy.pt').model.state_dict()
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
