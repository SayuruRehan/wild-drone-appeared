import copy
import itertools

import numpy as np
import pytest
import torch

from fleetrl.env import FleetEnv
from fleetrl.feasibility import feasible_mask
from fleetrl.features import MAX_WEIGHT, FeatureSpec, build_features, collate, snapshot
from fleetrl.models import KINDS, FleetQ
from fleetrl.selection import torch_dense_q
from fleetrl.train import LearnedPolicy, TrainConfig, train
from wild_drone.config import EnvConfig

CONFIG = EnvConfig(grid_size=16, horizon=40, battery_capacity=60, recharge_rate=5,
                   sensor_radius=1, inspection_deadline=12, max_fuel=40)
SPEC = FeatureSpec.from_config(CONFIG, local_radius=5, global_size=4)


def scattered_env(k, seed=0):
    env = FleetEnv(CONFIG, k)
    env.reset(seed)
    rng = np.random.default_rng(seed)
    env.positions = np.array([[7, 7], [8, 8], [3, 12], [12, 3], [5, 5], [9, 7]])[:k]
    env.ages = rng.integers(0, 20, env.ages.shape).astype(np.int32)
    return env, rng


def gain_of(env, commands):
    e = copy.deepcopy(env)
    _, _, info = e.step(np.array(commands))
    return info['reward_components']['freshness_gain']


def test_own_gain_and_overlap_are_the_reward_terms():
    env, _ = scattered_env(2)
    f = build_features(collate([snapshot(env)], 'cpu'), SPEC)
    wsum = float(env.priority_weights().sum())
    scale = (2*CONFIG.sensor_radius+1)**2*MAX_WEIGHT/wsum
    own = f['act'][0, :, :, 0].numpy()*scale
    overlap = f['overlap'][0, 0].numpy()*scale
    mask = feasible_mask(env).reshape(5, 5)
    assert mask.sum() > 10
    for a, b in itertools.product(range(5), repeat=2):
        if not mask[a, b]:
            continue  # rejected moves leave drones in place; never selected anyway
        expected = gain_of(env, [a, b])
        approx = own[0, a] + own[1, b] - overlap[a, b]
        assert approx == pytest.approx(expected, rel=2e-3, abs=1e-6)


FACTORED = [(k, v) for k in KINDS if k != 'joint' for v in ((1, 2) if k == 'dcg' else (2,))]


@pytest.mark.parametrize('kind,version', FACTORED)
def test_factored_models_are_permutation_equivariant_and_size_agnostic(kind, version):
    torch.manual_seed(0)
    model = FleetQ(kind, 2*SPEC.local_radius+1, SPEC.global_size, 32, model_version=version,
                   dcg_rank=2)
    for p in model.parameters():  # break the zero init so pairwise terms are active
        torch.nn.init.normal_(p, std=0.1)
    env, _ = scattered_env(3)
    f = build_features(collate([snapshot(env)], 'cpu'), SPEC)
    q = model.dense_q(f).view(5, 5, 5)
    perm = [2, 0, 1]
    env2 = copy.deepcopy(env)
    env2.positions = env.positions[perm]
    env2.batteries = env.batteries[perm]
    f2 = build_features(collate([snapshot(env2)], 'cpu'), SPEC)
    q2 = model.dense_q(f2).view(5, 5, 5)
    assert torch.allclose(q2, q.permute(*perm), atol=1e-4)
    for k in (1, 2, 5):
        env_k, _ = scattered_env(k)
        fk = build_features(collate([snapshot(env_k)], 'cpu'), SPEC)
        assert model.dense_q(fk).shape == (1, 5**k)


@pytest.mark.parametrize('kind,version', FACTORED+[('joint', 2)])
def test_q_of_matches_dense_table(kind, version):
    torch.manual_seed(1)
    model = FleetQ(kind, 2*SPEC.local_radius+1, SPEC.global_size, 32, n_drones=3,
                   model_version=version)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.1)
    env, _ = scattered_env(3)
    f = build_features(collate([snapshot(env), snapshot(env)], 'cpu'), SPEC)
    dense = model.dense_q(f)
    commands = torch.tensor([[0, 3, 4], [2, 2, 1]])
    flat = commands[:, 0]*25 + commands[:, 1]*5 + commands[:, 2]
    assert torch.allclose(model.q_of(f, commands), dense.gather(1, flat[:, None]).squeeze(1),
                          atol=1e-5)


def test_torch_dense_matches_numpy_order():
    unary = torch.randn(1, 3, 5)
    q = torch_dense_q(unary, None, [])
    assert q[0, 1*25+2*5+3] == pytest.approx(float(unary[0, 0, 1]+unary[0, 1, 2]+unary[0, 2, 3]),
                                             abs=1e-5)


def test_learned_policy_respects_mask_on_bigger_fleet():
    model = FleetQ('structured', 2*SPEC.local_radius+1, SPEC.global_size, 32)
    policy = LearnedPolicy(model, SPEC, 'cpu')
    env = FleetEnv(CONFIG, 6)
    env.reset(3)
    while not env.done:
        env.step(policy.act_commands(env))
    assert env.lost.sum() == 0 and env.totals['collision_attempts'] == 0


def test_training_smoke(tmp_path, monkeypatch):
    import fleetrl.train as T
    monkeypatch.setattr(T, 'env_config', lambda name: CONFIG)
    cfg = TrainConfig(kind='structured+', steps=300, warmup_steps=100, batch_size=8,
                      replay_size=1000, eval_every=150, eval_worlds=1, eval_drones=(3, 4),
                      local_radius=5, global_size=4, hidden=32, target_update_steps=50)
    out = train(cfg, tmp_path/'run')
    assert (out/'final.pt').exists() and (out/'done.json').exists()


def test_qmix_is_monotonic_in_each_utility():
    torch.manual_seed(2)
    model = FleetQ('qmix', 2*SPEC.local_radius+1, SPEC.global_size, 32)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.3)
    env, _ = scattered_env(3)
    f = build_features(collate([snapshot(env)], 'cpu'), SPEC)
    v, unary, mix = model.components(f)
    commands = torch.tensor([[1, 2, 3]])
    base = model.gather_from((v, unary, mix), f['pairs'], commands)
    bumped = unary.clone()
    bumped[0, 1, 2] += 1.0
    assert model.gather_from((v, bumped, mix), f['pairs'], commands) >= base


def test_ablations_use_only_their_feature():
    env, _ = scattered_env(3)
    f = build_features(collate([snapshot(env)], 'cpu'), SPEC)
    for kind, zeroed in (('structured-overlap', 'charger'), ('structured-charger', 'overlap')):
        torch.manual_seed(3)
        model = FleetQ(kind, 2*SPEC.local_radius+1, SPEC.global_size, 32)
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.1)
        q = model.dense_q(f)
        g = dict(f)
        g[zeroed] = torch.randn_like(f[zeroed])
        assert torch.allclose(q, model.dense_q(g))


def test_replay_roundtrip_and_sampling():
    from fleetrl.replay import Replay
    env, rng = scattered_env(3)
    buf = Replay(50, 3)
    ids = [buf.add_state(snapshot(env)) for _ in range(80)]
    for t in range(70):
        buf.add(ids[t], np.array([1, 2, 3]), float(t), 0.9, ids[t+3], np.ones(125, bool))
    assert len(buf) == 50
    restored = Replay.from_state_dict(buf.state_dict())
    a = buf.sample(np.random.default_rng(0), 8)
    b = restored.sample(np.random.default_rng(0), 8)
    for x, y in zip(a, b):
        if isinstance(x, dict):
            assert all(np.array_equal(x[k], y[k]) for k in x)
        else:
            assert np.array_equal(x, y)


def test_training_resumes_from_checkpoint(tmp_path, monkeypatch):
    import json

    import fleetrl.train as T
    monkeypatch.setattr(T, 'env_config', lambda name: CONFIG)
    cfg = dict(kind='qmix', steps=200, warmup_steps=40, batch_size=8, replay_size=1000,
               eval_every=100, eval_worlds=1, eval_drones=(3,), local_radius=5,
               global_size=4, hidden=32, target_update_steps=50, checkpoint_every=40,
               keep_checkpoint=True)
    out = train(TrainConfig(**cfg), tmp_path/'run')
    assert (out/'checkpoint.pt').exists()
    (out/'done.json').unlink()
    train(TrainConfig(**cfg), out)  # resumes from the last checkpoint and finishes
    steps = [json.loads(line)['steps'] for line in open(out/'log.jsonl')]
    assert steps == sorted(set(steps)) and steps[-1] == 200
    with pytest.raises(ValueError):
        train(TrainConfig(**{**cfg, 'learning_rate': 1e-3}), out)
