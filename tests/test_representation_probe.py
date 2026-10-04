import csv
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wild_drone.config import EnvConfig
from wild_drone.coverage_probe import CoverageConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.imitation_probe import collect_teacher_data
from wild_drone.learning import tensor_observations


def probe_module():
    assert importlib.util.find_spec('wild_drone.representation_probe'), 'representation missing'
    return importlib.import_module('wild_drone.representation_probe')


def test_relative_energy_features_match_physics_and_translate_with_base():
    probe = probe_module()
    cfg = EnvConfig(grid_size=8, horizon=6, battery_capacity=20, move_cost=2)
    env = WildfireFleetEnv(cfg)
    env.reset(seed=0)
    env.base = np.array([4, 4])
    env.positions = np.array([[2, 5], [4, 4], [0, 0]])
    env.batteries = np.array([12, 10, 0], dtype=np.float32)
    env.lost = np.array([False, False, True])
    _, vector = tensor_observations(env._observation())
    features = probe.relative_energy_features(vector, cfg).reshape(3, 4)
    torch.testing.assert_close(features[0], torch.tensor([-2/7, 1/7, 3/14, .3]))
    torch.testing.assert_close(features[1], torch.tensor([0., 0., 0., .5]))
    assert torch.count_nonzero(features[2]) == 0
    shifted = vector.clone()
    shifted[:, :12].reshape(-1, 3, 4)[:, :, :2] += 1/7
    shifted[:, 21:23] += 1/7
    torch.testing.assert_close(probe.relative_energy_features(shifted, cfg),
                               features.reshape(1, 12))
    vector[:, 2] = .1
    assert probe.relative_energy_features(vector, cfg)[0, 3] < 0  # Deficits must not be clipped.


def test_matched_networks_share_weights_and_control_padding_has_no_signal():
    probe = probe_module()
    cfg = EnvConfig(grid_size=8, horizon=6)
    models = []
    for arm in probe.ARMS:
        torch.manual_seed(0)
        models.append(probe.RepresentationNetwork(cfg, arm))
    a, b = (m.state_dict() for m in models)
    assert all(torch.equal(a[k], b[k]) for k in a)
    env = WildfireFleetEnv(cfg)
    obs, _ = env.reset(seed=5000)
    maps, vector = tensor_observations(obs)
    assert models[0](maps, vector)[0].shape == (1, 125)
    changed = vector.clone()
    changed[:, 2] /= 2
    assert torch.count_nonzero(models[0].extra_features(changed)) == 0
    assert not torch.equal(models[1].extra_features(changed), models[1].extra_features(vector))
    with pytest.raises(ValueError, match='arm'):
        probe.RepresentationNetwork(cfg, 'unknown')


def test_representation_exports_are_verified_and_validation_does_not_train(tmp_path):
    probe = probe_module()
    cfg = EnvConfig(grid_size=8, horizon=6)
    train = collect_teacher_data(cfg, [0, 1])
    val = collect_teacher_data(cfg, [5000])
    altered = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in val.items()}
    altered['actions'] = (altered['actions']+1) % 125
    settings = CoverageConfig(rounds=2, updates_per_round=4)
    for name, validation in [('a', val), ('b', altered)]:
        probe.fit_representation(cfg, train, validation, settings, 0, 'relative-energy',
                                 tmp_path/name)
    manifests = [json.loads((tmp_path/name/'checkpoint_manifest.json').read_text())
                 for name in ['a', 'b']]
    policies = [probe.load_representation(tmp_path/name/'final_representation.pt',
                                         manifest['final_representation.pt'])
                for name, manifest in zip(['a', 'b'], manifests)]
    states = [p.model.state_dict() for p in policies]
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])
    from wild_drone.learning import load_policy
    with pytest.raises(KeyError):
        load_policy(tmp_path/'a/final_representation.pt')
    with pytest.raises(ValueError, match='overlap'):
        probe.fit_representation(cfg, train, train, settings, 0, 'relative-energy',
                                 tmp_path/'overlap')
    with pytest.raises(ValueError, match='existing'):
        probe.fit_representation(cfg, train, val, settings, 0, 'relative-energy', tmp_path/'a')
    record = manifests[0]['final_representation.pt'] | {'model_sha256': 'wrong'}
    with pytest.raises(ValueError, match='identity'):
        probe.load_representation(tmp_path/'a/final_representation.pt', record)


def test_representation_runner_persists_all_missions_and_initial_pair_identity(tmp_path):
    path = Path(__file__).resolve().parents[1]/'scripts/run_representation_probe.py'
    assert path.exists(), 'matched representation runner missing'
    spec = importlib.util.spec_from_file_location('representation_runner', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    cfg = EnvConfig(grid_size=8, horizon=6)
    out = tmp_path/'run'
    runner.run_study(cfg, [0, 1], [5000], [0], [4, 8], 4, out)
    rows = list(csv.DictReader((out/'validation_missions.csv').open()))
    assert len(rows) == 7  # Two arms × (two budgets + selected), plus one Greedy.
    assert rows[-1]['policy'] == 'greedy'
    a, b = [json.loads((out/arm/'seed-0/checkpoint_manifest.json').read_text())
            for arm in ['padded-control', 'relative-energy']]
    assert a['round-0.pt']['model_sha256'] == b['round-0.pt']['model_sha256']
    protocol = json.loads((out/'protocol.json').read_text())
    assert protocol['expected_missions'] == len(rows)
    assert protocol['matched_parameter_count'] > 0
