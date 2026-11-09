"""The N-drone code must reproduce the frozen assignment code exactly at N=3."""

import json
from pathlib import Path

import numpy as np
import pytest

from fleetrl.baselines import GreedyPolicy
from fleetrl.env import FleetEnv
from fleetrl.feasibility import decode, encode, feasible_mask
from wild_drone.baselines import GreedyPolicy as FrozenGreedy
from wild_drone.baselines import candidate_actions
from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {
    'dev20': EnvConfig(**json.load(open(ROOT/'configs/development.json'))['environment']),
    'main40': EnvConfig(**json.load(open(ROOT/'configs/rl-main-frozen.json'))['environment']),
}


def frozen_mask(env):
    mask = np.zeros(125, dtype=bool)
    mask[candidate_actions(env)[0]] = True
    return mask


@pytest.mark.parametrize('name', CONFIGS)
@pytest.mark.parametrize('seed', [3, 97])
def test_trajectories_mask_and_rewards_match(name, seed):
    config = CONFIGS[name]
    frozen, ours = WildfireFleetEnv(config), FleetEnv(config, 3)
    frozen.reset(seed=seed)
    ours.reset(seed)
    rng = np.random.default_rng(seed)
    done = False
    steps = 0
    while not done:
        mask = frozen_mask(frozen)
        assert np.array_equal(mask, feasible_mask(ours, strict=True)), f'mask differs at step {steps}'
        # mostly feasible actions, sometimes arbitrary ones to exercise collision rules
        action = (int(rng.choice(np.flatnonzero(mask))) if rng.random() < 0.8
                  else int(rng.integers(125)))
        _, r1, d1, _, i1 = frozen.step(action)
        r2, d2, i2 = ours.step(decode(action, 3))
        assert r1 == pytest.approx(r2, abs=1e-12)
        assert d1 == d2
        for key in ('weighted_freshness', 'overdue_fraction', 'burning_cells'):
            assert i1[key] == pytest.approx(i2[key], abs=1e-12)
        assert np.array_equal(frozen.positions, ours.positions)
        assert np.array_equal(frozen.batteries, ours.batteries)
        assert np.array_equal(frozen.ages, ours.ages)
        assert np.array_equal(frozen.burning, ours.burning)
        assert frozen.charger_owner == ours.charger_owner
        done = d1
        steps += 1
        if steps >= 120:
            break


@pytest.mark.parametrize('seed', [0, 11])
def test_greedy_matches_frozen_greedy(seed):
    config = CONFIGS['main40']
    frozen, ours = WildfireFleetEnv(config), FleetEnv(config, 3)
    frozen.reset(seed=seed)
    ours.reset(seed)
    g_frozen, g_ours = FrozenGreedy(), GreedyPolicy()
    for _ in range(80):
        a = g_frozen.act(frozen, None)
        b = g_ours.act_commands(ours)
        assert a == encode(b)
        frozen.step(a)
        ours.step(b)


def test_encode_decode_roundtrip():
    for n in (1, 3, 5):
        for index in sorted({0, min(7, 5**n-1), 5**n-1}):
            assert encode(decode(index, n)) == index
