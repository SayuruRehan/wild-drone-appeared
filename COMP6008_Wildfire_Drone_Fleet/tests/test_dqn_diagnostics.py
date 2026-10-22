import pytest
import torch

from wild_drone.config import EnvConfig
from wild_drone.diagnostics import DQNDiagnosticPolicy, discounted_upper_bound
from wild_drone.environment import WildfireFleetEnv
from wild_drone.learning import FleetNetwork, NeuralPolicy, TrainConfig


@pytest.mark.parametrize('gamma,expected', [(0, 1), (0.5, 1.75), (1, 3)])
def test_discounted_upper_bound_handles_endpoints(gamma, expected):
    assert discounted_upper_bound(3, gamma, 1) == pytest.approx(expected)
    assert discounted_upper_bound(0, gamma, 1) == 0


def test_diagnostic_policy_preserves_actions_and_tracks_remaining_horizon_bound():
    config = EnvConfig(grid_size=8, horizon=3)
    model = FleetNetwork(8)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head.bias[0] = 2
    policy = NeuralPolicy(model, 'dqn', config, TrainConfig(gamma=1))
    diagnostic = DQNDiagnosticPolicy(policy)
    env = WildfireFleetEnv(config)
    obs, _ = env.reset(seed=5000)
    assert diagnostic.act(env, obs) == policy.act(env, obs)
    assert diagnostic.violations == 0  # Q=2 <= three remaining rewards of at most one.
    env.elapsed = 2
    diagnostic.act(env, env._observation())
    assert diagnostic.violations == 1
    assert diagnostic.max_excess == 1 and diagnostic.max_q == 2
    assert diagnostic.decisions == 2
