import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wild_drone.baselines import GreedyPolicy
from wild_drone.config import EnvConfig
from wild_drone.coverage_probe import CoverageConfig, fit_coverage
from wild_drone.environment import WildfireFleetEnv
from wild_drone.imitation_probe import collect_teacher_data, data_digest


def module():
    assert importlib.util.find_spec('wild_drone.recovery_probe'), 'recovery collection missing'
    return importlib.import_module('wild_drone.recovery_probe')


class NorthFirstDrone:
    def act(self, env, observation):
        return 25


def test_assistance_gates_real_collection_and_all_labels_replay_without_rescue():
    probe = module()
    cfg = EnvConfig(grid_size=8, horizon=40, battery_capacity=10)
    records = []
    for arm in probe.ARMS:
        data, audit = probe.collect_recovery_rollouts(cfg, [0], NorthFirstDrone(), arm, 7,
                                                       teacher_probability=0)
        env = WildfireFleetEnv(cfg)
        env.reset(seed=0)
        for index, row in enumerate(audit):
            np.testing.assert_array_equal(data['positions'][index], env.positions)
            from wild_drone.learning import tensor_observations
            maps, vector = tensor_observations(env._observation())
            torch.testing.assert_close(data['maps'][index], maps[0], rtol=0, atol=0)
            torch.testing.assert_close(data['vector'][index], vector[0], rtol=0, atol=0)
            assert int(data['steps'][index]) == env.elapsed == row['step']
            assert row['teacher_action'] == GreedyPolicy().act(env, env._observation())
            assert row['learner_action'] == 25
            if arm == 'learner':
                assert not row['teacher_execution'] and row['executed_action'] == 25
            elif row['risk_gate'] or row['charging_gate']:
                assert row['teacher_execution']
                assert row['executed_action'] == row['teacher_action']
            env.step(row['executed_action'])
        assert env.elapsed == cfg.horizon or env.lost.all()
        records.append(audit)
    assert any(row['risk_gate'] for row in records[1])
    assert any(row['charging_gate'] for row in records[1])
    assert sum(row['new_charge_completions'] for row in records[1]) > 0


def test_collection_randomness_is_reproducible_and_configuration_rejected():
    probe = module()
    cfg = EnvConfig(grid_size=8, horizon=20, battery_capacity=10)
    a, b = [probe.collect_recovery_rollouts(cfg, [0, 1], NorthFirstDrone(), 'assisted', 19)
            for _ in range(2)]
    assert data_digest(a[0]) == data_digest(b[0]) and a[1] == b[1]
    for kwargs in [{'teacher_probability': -1}, {'return_margin_move_costs': -1}]:
        with pytest.raises(ValueError):
            probe.collect_recovery_rollouts(cfg, [0], NorthFirstDrone(), 'assisted', 0, **kwargs)


def test_first_round_matches_original_fixed_fitter_and_deployment_is_learner_only(tmp_path):
    probe = module()
    cfg = EnvConfig(grid_size=8, horizon=6)
    initial = collect_teacher_data(cfg, [0, 1])
    val = collect_teacher_data(cfg, [5000])
    settings = CoverageConfig(rounds=2, updates_per_round=4)
    fit_coverage(cfg, initial, val, [], settings, 0, 'fixed', tmp_path/'fixed')
    for arm in probe.ARMS:
        result = probe.fit_recovery(cfg, initial, val, [[0]], settings, 0, arm, tmp_path/arm)
        assert result['completed_updates'] == 8
        assert (tmp_path/arm/'added-round-2.pt').exists()
    manifests = [json.loads((tmp_path/name/'checkpoint_manifest.json').read_text())
                 for name in ['fixed', *probe.ARMS]]
    assert len({m['round-1.pt']['model_sha256'] for m in manifests}) == 1
    manifest = manifests[-1]
    policy = probe.load_recovery(tmp_path/'assisted/final_recovery.pt',
                                 manifest['final_recovery.pt'])
    env = WildfireFleetEnv(cfg)
    obs, _ = env.reset(seed=5000)
    with torch.no_grad():
        from wild_drone.learning import tensor_observations
        expected = int(policy.model(*tensor_observations(obs))[0].argmax(1))
    assert policy.act(env, obs) == expected  # No collection gate or teacher in deployment.
    from wild_drone.learning import load_policy
    with pytest.raises(KeyError):
        load_policy(tmp_path/'assisted/final_recovery.pt')
    with pytest.raises(ValueError, match='overlap'):
        probe.fit_recovery(cfg, initial, val, [[5000]], settings, 0, 'assisted', tmp_path/'bad')


def test_validation_selection_does_not_change_training_or_added_data(tmp_path):
    probe = module()
    cfg = EnvConfig(grid_size=8, horizon=6)
    initial = collect_teacher_data(cfg, [0, 1])
    val = collect_teacher_data(cfg, [5000])
    altered = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in val.items()}
    altered['actions'] = (altered['actions']+1) % 125
    settings = CoverageConfig(rounds=2, updates_per_round=4)
    for name, data in [('a', val), ('b', altered)]:
        probe.fit_recovery(cfg, initial, data, [[0]], settings, 0, 'assisted', tmp_path/name)
    manifests = [json.loads((tmp_path/name/'checkpoint_manifest.json').read_text())
                 for name in ['a', 'b']]
    assert manifests[0]['final_recovery.pt']['model_sha256'] == \
        manifests[1]['final_recovery.pt']['model_sha256']
    added = [torch.load(tmp_path/name/'added-round-2.pt', weights_only=True) for name in ['a', 'b']]
    assert data_digest(added[0]) == data_digest(added[1])


def test_recovery_runner_binds_initial_prefix_and_all_persisted_missions(tmp_path):
    path = Path(__file__).resolve().parents[1]/'scripts/run_recovery_probe.py'
    assert path.exists(), 'recovery runner missing'
    spec = importlib.util.spec_from_file_location('recovery_runner', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    cfg = EnvConfig(grid_size=8, horizon=6)
    out = tmp_path/'run'
    settings = CoverageConfig(rounds=2, updates_per_round=4)
    runner.run_study(cfg, [0, 1], [[0, 1]], [5000], [0], settings, out)
    complete = json.loads((out/'completion.json').read_text())
    assert complete['persisted_missions'] == 7
    assert len(set(complete['paired_first_round_model_sha256'].values())) == 1
    with pytest.raises(ValueError, match='existing'):
        runner.run_study(cfg, [0, 1], [[0, 1]], [5000], [0], settings, out)
