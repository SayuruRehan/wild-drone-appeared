import copy
import json

import numpy as np
import pytest

from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.masked_learning import action_mask


def probe():
    from wild_drone import ranking_probe
    return ranking_probe


def scene():
    env = WildfireFleetEnv(EnvConfig(grid_size=8, horizon=40, sensor_radius=1))
    env.reset(seed=5900)
    env.positions[:] = [[2, 2], [2, 5], [5, 2]]
    env.batteries[:] = 100
    return env


def test_shared_score_shift_changes_values_without_changing_rank_or_top_action():
    scores = np.arange(125, dtype=float)
    mask = np.ones(125, dtype=bool)
    row = probe().pair_metrics(scores, scores+20, scores, scores, mask, mask)
    assert row['score_delta_max'] == 20
    assert row['centred_score_delta_max'] == pytest.approx(0)
    assert row['order_change_fraction'] == 0
    assert not row['common_top_changed']


def test_mask_change_is_separate_from_a_learned_ranking_change():
    scores = -np.arange(125, dtype=float)
    before = np.zeros(125, dtype=bool)
    after = before.copy()
    before[[0, 1, 2]] = True
    after[[1, 2, 3]] = True
    row = probe().pair_metrics(scores, scores, scores, scores, before, after)
    assert row['common_actions'] == 2
    assert row['deployment_top_changed']
    assert not row['common_top_changed']
    assert row['order_change_fraction'] == 0


def test_rank_reversal_and_ties_are_counted_and_one_step_regret_is_normalized():
    before = np.zeros(125)
    after = before.copy()
    before[:3] = [0, 1, 1]
    after[:3] = [2, 1, 0]
    mask = np.zeros(125, dtype=bool)
    mask[:3] = True
    row = probe().pair_metrics(before, after, before, after, mask, mask)
    assert row['order_change_fraction'] == 1
    assert row['strict_flip_fraction'] == 1
    assert row['common_top_changed']
    assert row['before_one_step_optimal'] and row['after_one_step_optimal']
    rank = probe().ranking_metrics(before, after, np.flatnonzero(mask))
    assert rank['selected_action'] == 1  # stable tie: first maximum
    assert rank['normalised_gain_regret'] == .5
    assert rank['reference_best_count'] == 1
    assert rank['feasible_actions'] == 3
    assert rank['random_best_probability'] == pytest.approx(1/3)
    assert probe().ranking_metrics(before, np.ones(125), np.flatnonzero(mask))[
        'normalised_gain_regret'] is None


def test_transition_coverage_matches_real_inspection_rewards_without_state_or_rng_mutation():
    env = scene()
    env.ages[:] = np.arange(64).reshape(8, 8)
    before = copy.deepcopy(env)
    coverage = probe().transition_coverage(env)
    gains = probe().gain_scores(env, coverage)
    for action in range(125):
        clone = copy.deepcopy(before)
        info = clone.step(action)[4]
        assert gains[action] == pytest.approx(info['reward_components']['freshness_gain'])
    for name in ['ages', 'positions', 'batteries', 'burning', 'fuel', 'terrain', 'priority']:
        np.testing.assert_array_equal(getattr(env, name), getattr(before, name))
    assert env.elapsed == before.elapsed and env.totals == before.totals
    assert json.dumps(env.np_random.bit_generator.state) == json.dumps(
        before.np_random.bit_generator.state)


def test_urgency_pair_changes_only_ages_with_identical_masks_and_disjoint_best_actions():
    env = scene()
    coverage = probe().transition_coverage(env)
    result = probe().urgency_pair(env, coverage)
    assert result is not None
    a, b, detail = result
    assert detail['kind'] == 'urgency'
    for other in [a, b]:
        np.testing.assert_array_equal(other.positions, env.positions)
        np.testing.assert_array_equal(other.batteries, env.batteries)
        np.testing.assert_array_equal(other.burning, env.burning)
        np.testing.assert_array_equal(action_mask(other), action_mask(env))
    ids = np.flatnonzero(action_mask(env))
    ga, gb = [probe().gain_scores(state, coverage) for state in [a, b]]
    np.testing.assert_array_equal(np.sort(a.ages.ravel()), np.sort(b.ages.ravel()))
    assert float((env.priority_weights()*a.ages).sum()) == float(
        (env.priority_weights()*b.ages).sum())
    assert not set(probe().best_actions(ga, ids)) & set(probe().best_actions(gb, ids))


def test_position_pair_changes_only_one_live_position_and_has_shared_feasible_control():
    env = scene()
    env.ages[:] = np.indices((8, 8)).sum(axis=0)*3
    coverage = probe().transition_coverage(env)
    result = probe().position_pair(env, coverage)
    assert result is not None
    after, changed_coverage, detail = result
    assert detail['kind'] == 'position'
    assert np.abs(env.positions-after.positions).sum() == 1
    for name in ['ages', 'batteries', 'burning', 'fuel', 'terrain', 'priority']:
        np.testing.assert_array_equal(getattr(env, name), getattr(after, name))
    common = np.flatnonzero(action_mask(env) & action_mask(after))
    assert len(common) >= 2
    ga, gb = probe().gain_scores(env, coverage), probe().gain_scores(after, changed_coverage)
    assert not set(probe().best_actions(ga, common)) & set(probe().best_actions(gb, common))


def test_invalid_score_vectors_are_rejected():
    mask = np.ones(125, dtype=bool)
    for scores in [np.zeros(124), np.full(125, np.nan)]:
        with pytest.raises(ValueError, match='finite.*125'):
            probe().pair_metrics(scores, np.zeros(125), np.zeros(125), np.zeros(125), mask, mask)


def test_network_scores_use_current_observation_and_preserve_deployed_argmax():
    import torch

    from wild_drone.learning import FleetNetwork, TrainConfig
    from wild_drone.masked_learning import MaskedNeuralPolicy

    torch.set_num_threads(1)
    torch.manual_seed(101)
    env = scene()
    policy = MaskedNeuralPolicy(FleetNetwork(8), 'dqn', env.config,
                               train_config=TrainConfig())
    before = probe().network_scores(policy, env)
    assert np.flatnonzero(action_mask(env))[np.argmax(before[action_mask(env)])] == policy.act(
        env, env._observation())
    np.testing.assert_array_equal(before, probe().network_scores(policy, copy.deepcopy(env)))
    env.ages[:] = 0
    env.positions[0, 0] += 1
    after = probe().network_scores(policy, env)
    assert not np.array_equal(before, after)
