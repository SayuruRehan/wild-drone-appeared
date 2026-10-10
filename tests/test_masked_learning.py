import copy

import numpy as np
import pytest
import torch

from wild_drone.baselines import candidate_actions
from wild_drone.config import EnvConfig
from wild_drone.environment import MOVES, WildfireFleetEnv, decode_action
from wild_drone.experiments import run_episode
from wild_drone.learning import TrainConfig
from wild_drone.masked_learning import (
    MaskedRandomPolicy,
    action_mask,
    apply_mask,
    load_masked_policy,
    masked_max,
    train_masked,
)

# Small battery: an unmasked random fleet strands drones within this horizon.
TIGHT = EnvConfig(grid_size=8, horizon=60, battery_capacity=12, recharge_rate=4)


def test_mask_is_exactly_the_baseline_candidate_set():
    env = WildfireFleetEnv(TIGHT)
    env.reset(seed=3)
    for _ in range(20):
        mask = action_mask(env)
        assert mask.dtype == bool and mask.shape == (125,)
        np.testing.assert_array_equal(np.flatnonzero(mask), np.sort(candidate_actions(env)[0]))
        env.step(int(np.flatnonzero(mask)[-1]))


def deadlocked_env():
    """Observed failure: three low-battery drones whose only base-ward cells collide."""
    env = WildfireFleetEnv(EnvConfig(grid_size=20, horizon=160, battery_capacity=80))
    env.reset(seed=5000)
    env.positions = np.array([[9, 9], [8, 10], [10, 8]], dtype=np.int32)
    env.batteries = np.array([3.5, 2.0, 3.5], dtype=np.float32)
    return env


def test_baseline_rule_deadlocks_into_hover_but_mask_returns_drones_home():
    env = deadlocked_env()
    assert candidate_actions(env)[0].tolist() == [0]  # the rule's all-stay fallback
    mask = action_mask(env)
    assert not mask[0]
    distance = np.abs(env.positions-env.base).sum(axis=1)
    for action in np.flatnonzero(mask):
        commands = decode_action(int(action))
        moved = env.positions + MOVES[commands]
        closer = np.abs(moved-env.base).sum(axis=1) < distance
        assert closer[1]  # drone 1 has no battery margin left and must move now
        assert np.all(closer | (commands == 0))  # others move home or wait (hover is cheaper)
        assert len({tuple(p) for p in moved}) == 3  # no blocked, energy-wasting proposals
        probe = copy.deepcopy(env)
        assert probe.step(int(action))[4]['collision_attempts'] == 0
    policy = MaskedRandomPolicy(0)
    for _ in range(20):
        env.step(policy.act(env, None))
    assert env.lost.sum() == 0


def test_masked_actions_get_zero_probability_and_never_win_argmax():
    logits = torch.zeros(2, 125)
    logits[:, 7] = 50.0  # the best raw logit is infeasible
    mask = np.zeros((2, 125), dtype=bool)
    mask[:, [1, 2]] = True
    masked = apply_mask(logits, mask)
    probs = torch.softmax(masked, dim=1)
    assert torch.all(probs[:, ~torch.as_tensor(mask[0])] == 0)
    assert set(masked.argmax(dim=1).tolist()) <= {1, 2}
    entropy = torch.distributions.Categorical(logits=masked).entropy()
    assert torch.all(torch.isfinite(entropy))


def test_masked_max_ignores_infeasible_next_actions():
    q = torch.tensor([[1., 9., 3.], [4., 2., 8.]])
    mask = np.array([[True, False, True], [True, True, False]])
    torch.testing.assert_close(masked_max(q, mask), torch.tensor([3., 4.]))


def test_unmasked_random_strands_drones_but_masked_random_does_not():
    from wild_drone.baselines import RandomPolicy
    unmasked = [run_episode(TIGHT, RandomPolicy(s), s, name='random')['total_losses']
                for s in range(3)]
    masked = [run_episode(TIGHT, MaskedRandomPolicy(s), s, name='masked_random')['total_losses']
              for s in range(3)]
    assert sum(unmasked) > 0
    assert masked == [0, 0, 0]


@pytest.mark.parametrize('algorithm', ['dqn', 'reinforce', 'a2c'])
def test_masked_training_roundtrip_and_feasible_actions(algorithm, tmp_path):
    training = TrainConfig(episodes=3, seed=1, batch_size=4, warmup_steps=4,
                           target_update_steps=4, n_steps=4, validation_seeds=(5000,),
                           validation_interval=2)
    summary = train_masked(algorithm, TIGHT, training, tmp_path)
    assert summary['updates'] > 0 and np.isfinite(summary['last_loss'])
    assert (tmp_path/'training.csv').exists() and (tmp_path/'validation.csv').exists()
    for name in ['policy.pt', 'best_policy.pt']:
        policy = load_masked_policy(tmp_path/name)
        env = WildfireFleetEnv(TIGHT)
        observation, _ = env.reset(seed=5001)
        done = False
        while not done:
            action = policy.act(env, observation)
            assert action_mask(env)[action]
            observation, _, done, _, _ = env.step(action)
        assert env.lost.sum() == 0


def test_masked_training_refuses_to_overwrite(tmp_path):
    training = TrainConfig(episodes=1, batch_size=4, warmup_steps=4, n_steps=4)
    train_masked('a2c', TIGHT, training, tmp_path)
    with pytest.raises(ValueError, match='existing'):
        train_masked('a2c', TIGHT, training, tmp_path)
