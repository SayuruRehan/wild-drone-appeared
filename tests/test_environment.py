import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv, decode_action, encode_action


def make_env(**kwargs):
    env = WildfireFleetEnv(EnvConfig(grid_size=8, horizon=40, **kwargs))
    env.reset(seed=17)
    return env


def test_joint_actions_roundtrip():
    assert [encode_action(decode_action(a)) for a in range(125)] == list(range(125))
    with pytest.raises(ValueError):
        decode_action(125)


def test_gymnasium_contract():
    check_env(make_env(), skip_render_check=True)


def test_seeded_dynamics_and_complete_observation():
    a, b = make_env(), make_env()
    for action in [0, 21, 49, 62]:
        oa, ra, ta, _, _ = a.step(action)
        ob, rb, tb, _, _ = b.step(action)
        for key in oa:
            np.testing.assert_array_equal(oa[key], ob[key])
        assert (ra, ta) == (rb, tb)
        assert a.observation_space.contains(oa)
    old = oa['maps'].copy()
    a.step(0)
    np.testing.assert_array_equal(oa['maps'], old)


def test_short_mission_unvisited_cells_start_stale_and_become_overdue():
    env = WildfireFleetEnv(EnvConfig(grid_size=8, horizon=6, inspection_deadline=30))
    obs, info = env.reset(seed=4)
    assert info['weighted_freshness'] == 0
    _, _, _, _, info = env.step(0)
    assert env.ages[0, 0] > 30
    assert info['overdue_fraction'] > 0
    assert env.observation_space.contains(env._observation())


def test_same_destination_is_blocked():
    env = make_env()
    env.positions[:] = [[2, 1], [2, 3], [6, 6]]
    _, _, _, _, info = env.step(encode_action([4, 3, 0]))
    np.testing.assert_array_equal(env.positions[:2], [[2, 1], [2, 3]])
    assert info['collision_attempts'] == 2


def test_swap_and_following_collision_cascade():
    env = make_env()
    env.positions[:] = [[2, 1], [2, 2], [2, 3]]
    env.step(encode_action([4, 3, 3]))
    np.testing.assert_array_equal(env.positions, [[2, 1], [2, 2], [2, 3]])


def test_boundary_move_stays_in_bounds():
    env = make_env()
    env.positions[0] = [0, 0]
    env.step(encode_action([1, 0, 0]))
    np.testing.assert_array_equal(env.positions[0], [0, 0])


def test_base_allows_parking_but_one_charger():
    env = make_env()
    env.batteries[:] = [10, 20, 30]
    obs, _, _, _, _ = env.step(0)
    np.testing.assert_array_equal(env.batteries, [18, 20, 30])
    assert env.charger_owner == 0
    assert obs['globals'][5:9].argmax() == 1
    # An incumbent keeps the charger even if another drone is lower now.
    env.batteries[1] = 1
    env.step(0)
    assert env.batteries[0] == 26
    assert env.batteries[1] == 1


def test_arrival_cannot_charge_until_next_step_and_zero_arrival_is_safe():
    env = make_env()
    env.positions[0] = env.base + [0, 1]
    env.batteries[0] = 1
    _, _, _, _, info = env.step(encode_action([3, 0, 0]))
    assert env.batteries[0] == 0
    assert not env.lost[0]
    assert info['safe_returns'] == 1
    env.step(0)
    assert env.batteries[0] == 8


def test_depletion_away_from_base_is_permanent_and_cannot_inspect():
    env = make_env(sensor_radius=0)
    env.positions[0] = [1, 1]
    env.batteries[0] = 0.25
    _, _, _, _, info = env.step(0)
    assert env.lost[0]
    assert env.ages[1, 1] > env.config.inspection_deadline
    assert info['new_losses'] == 1
    pos = env.positions[0].copy()
    env.step(encode_action([4, 0, 0]))
    np.testing.assert_array_equal(env.positions[0], pos)


def test_overlapping_scans_count_once():
    env = make_env(sensor_radius=1)
    _, _, _, _, info = env.step(0)
    assert info['inspected_cells'] == 9
    assert np.count_nonzero(env.ages == 0) == 9


def test_fire_consumes_fuel_and_spread_has_no_edge_wrap():
    env = make_env(spread_probability=1, wind_change_probability=0)
    env.burning[:] = False
    env.burning[0, 0] = True
    env.fuel[:] = env.config.max_fuel
    env.terrain[:] = 1
    env.fuel[0, 0] = 1
    env.wind = 0
    env.step(0)
    assert not env.burning[0, 0]
    assert env.burning[0, 1] and env.burning[1, 0]
    assert not env.burning[-1, 0]
    assert not env.burning[0, -1]
    assert env.fuel[0, 1] == env.config.max_fuel


def test_terminal_mission_horizon_and_no_step_after_done():
    env = make_env()
    for _ in range(40):
        _, _, terminated, truncated, _ = env.step(0)
    assert terminated and not truncated
    with pytest.raises(RuntimeError):
        env.step(0)


def test_all_drones_lost_ends_episode():
    env = make_env()
    env.positions[:] = [[1, 1], [2, 2], [3, 3]]
    env.batteries[:] = 0.25
    _, _, terminated, _, info = env.step(0)
    assert terminated and info['total_losses'] == 3


def test_fleet_loss_charges_remaining_overdue_mission_cost():
    env = make_env()
    env.positions[:] = [[1, 1], [2, 2], [3, 3]]
    env.batteries[:] = 0.25
    _, _, _, _, info = env.step(0)
    assert info['reward_components']['mission_failure'] == pytest.approx(-3.9)


def test_invalid_action_does_not_advance_state():
    env = make_env()
    with pytest.raises(ValueError):
        env.step(-1)
    assert env.elapsed == 0


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        EnvConfig(grid_size=2)
    with pytest.raises(ValueError):
        EnvConfig(spread_probability=1.1)
