"""Behaviour for fleet sizes other than three."""

import itertools

import numpy as np
import pytest

from fleetrl.baselines import AssignPolicy, GreedyPolicy, MaskedRandomPolicy
from fleetrl.env import FleetEnv
from fleetrl.evaluate import run_episode
from fleetrl.feasibility import feasible_mask, unary_allowed
from fleetrl.selection import dense_scores, select
from wild_drone.config import EnvConfig

SMALL = EnvConfig(grid_size=12, horizon=60, battery_capacity=40, recharge_rate=5,
                  sensor_radius=1, inspection_deadline=15, max_fuel=40)


def brute_force_mask(env):
    """Reference: apply each joint action to a copy and check the rules directly."""
    k = env.n_drones
    allowed = unary_allowed(env)
    out = []
    for commands in itertools.product(range(5), repeat=k):
        ok = all(allowed[i, a] for i, a in enumerate(commands))
        proposed = env.positions + np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])[
            list(commands)]
        for i, j in itertools.combinations(range(k), 2):
            if env.lost[i] or env.lost[j]:
                continue
            same = np.array_equal(proposed[i], proposed[j]) and not np.array_equal(
                proposed[i], env.base)
            swap = (np.array_equal(proposed[i], env.positions[j]) and
                    np.array_equal(proposed[j], env.positions[i]) and
                    not np.array_equal(env.positions[i], env.positions[j]))
            ok &= not (same or swap)
        out.append(ok)
    return np.array(out)


@pytest.mark.parametrize('k', [1, 2, 4])
def test_factored_mask_matches_brute_force(k):
    env = FleetEnv(SMALL, k)
    env.reset(5)
    rng = np.random.default_rng(0)
    for _ in range(40):
        mask = feasible_mask(env)
        reference = brute_force_mask(env)
        if reference.any():
            assert np.array_equal(mask, reference)
        env.step(np.array(np.unravel_index(int(rng.choice(np.flatnonzero(mask))),
                                           (5,)*k)))
        if env.done:
            break


@pytest.mark.parametrize('k', [2, 5, 7])
def test_masked_policies_never_collide_or_lose_drones(k):
    for policy in (MaskedRandomPolicy(1), GreedyPolicy(), AssignPolicy()):
        result = run_episode(SMALL, k, policy, seed=2)
        assert result['losses'] == 0
        assert result['collision_attempts'] == 0


def test_dense_selection_is_exact():
    rng = np.random.default_rng(0)
    k = 4
    unary = rng.normal(size=(k, 5))
    pair = rng.normal(size=(k, k, 5, 5))
    mask = rng.random(5**k) < 0.3
    index = select(unary, pair, mask)
    best, best_index = -np.inf, None
    for flat, commands in enumerate(itertools.product(range(5), repeat=k)):
        if not mask[flat]:
            continue
        value = sum(unary[i, a] for i, a in enumerate(commands))
        value += sum(pair[i, j, commands[i], commands[j]]
                     for i, j in itertools.combinations(range(k), 2))
        if value > best:
            best, best_index = value, flat
    assert index == best_index
    assert dense_scores(unary, pair).reshape(-1)[index] == pytest.approx(best)


def test_more_drones_share_one_charger():
    env = FleetEnv(SMALL, 6)
    env.reset(0)
    policy = GreedyPolicy()
    while not env.done:
        env.step(policy.act_commands(env))
    assert env.totals['charge_completions'] > 0
    assert env.totals['charger_wait_steps'] > 0


def test_pinwheel_deadlock_is_resolved_without_losses():
    """Regression: three low-battery drones around the pad blocked each other completely
    under the frozen rules and all were lost (40x40 pilot, world 30008)."""
    from fleetrl.feasibility import constraints
    from wild_drone.config import EnvConfig
    config = EnvConfig()  # 40x40 frozen settings, base (20, 20)
    env = FleetEnv(config, 3)
    env.reset(30008)
    env.positions = np.array([[22, 20], [20, 22], [21, 21]])
    env.batteries = np.array([3.0, 3.0, 3.0], dtype=np.float32)
    assert not feasible_mask(env, strict=True)[1:].any()      # frozen: only "all stay"
    assert constraints(env)[2] == 'relaxed'
    policy = GreedyPolicy()
    for _ in range(6):
        env.step(policy.act_commands(env))
    assert env.lost.sum() == 0
    assert np.all(np.abs(env.positions-env.base).sum(axis=1) == 0)
