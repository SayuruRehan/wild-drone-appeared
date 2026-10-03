import numpy as np
import pytest
import torch

from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.learning import (
    TrainConfig,
    discounted_policy_loss,
    discounted_returns,
    load_policy,
    td_targets,
    train,
)


def test_monte_carlo_returns_and_bootstrap():
    np.testing.assert_allclose(discounted_returns([1, 2, 3], 0.5), [2.75, 3.5, 3])
    np.testing.assert_allclose(discounted_returns([1, 2], 0.5, bootstrap=4), [3, 4])


def test_terminal_td_target_does_not_bootstrap():
    targets = td_targets(torch.tensor([1., 1.]), torch.tensor([True, False]),
                         torch.tensor([100., 2.]), 0.5)
    torch.testing.assert_close(targets, torch.tensor([1., 2.]))


def test_policy_gradient_weights_absolute_discount_and_fixed_horizon():
    log_probs = torch.tensor([-1., -1.], requires_grad=True)
    returns = torch.tensor([1., 1.])
    loss = discounted_policy_loss(log_probs, returns, 0.5, start_step=0, normalizer=4)
    loss.backward()
    assert loss.item() == pytest.approx(0.375)
    torch.testing.assert_close(log_probs.grad, torch.tensor([-0.25, -0.125]))
    later = discounted_policy_loss(log_probs, returns, 0.5, start_step=2, normalizer=4)
    assert later.item() == pytest.approx(0.09375)


@pytest.mark.parametrize('algorithm', ['dqn', 'reinforce', 'a2c'])
def test_training_updates_and_checkpoint_roundtrip(algorithm, tmp_path):
    config = EnvConfig(grid_size=8, horizon=8)
    training = TrainConfig(episodes=2, seed=3, batch_size=4, warmup_steps=4,
                           target_update_steps=4, n_steps=4)
    report = train(algorithm, config, training, tmp_path / algorithm)
    assert report['updates'] > 0
    assert np.isfinite(report['last_loss'])
    policy = load_policy(tmp_path / algorithm / 'policy.pt')
    env = WildfireFleetEnv(config)
    obs, _ = env.reset(seed=10000)
    action = policy.act(env, obs)
    assert env.action_space.contains(action)
    duplicate = load_policy(tmp_path / algorithm / 'policy.pt')
    assert duplicate.act(env, obs) == action


def test_training_seed_reproduces_small_run(tmp_path):
    cfg = EnvConfig(grid_size=8, horizon=6)
    settings = TrainConfig(episodes=2, seed=5, n_steps=3)
    a = train('a2c', cfg, settings, tmp_path / 'a')
    b = train('a2c', cfg, settings, tmp_path / 'b')
    assert a['episode_returns'] == b['episode_returns']
    assert a['last_loss'] == b['last_loss']


def test_invalid_training_settings_fail_early():
    with pytest.raises(ValueError):
        TrainConfig(episodes=0)
    with pytest.raises(ValueError):
        TrainConfig(gamma=2)
