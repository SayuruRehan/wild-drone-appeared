import json
from dataclasses import replace

import pytest
import torch

from wild_drone.config import EnvConfig
from wild_drone.coverage_probe import (
    CoverageConfig,
    collect_rollouts,
    fit_coverage,
    load_coverage,
    merge_data,
)
from wild_drone.environment import WildfireFleetEnv
from wild_drone.imitation_probe import collect_teacher_data, data_digest, load_probe
from wild_drone.learning import load_policy, tensor_observations


class Park:
    def act(self, env, observation):
        return 0


def test_learner_rollouts_store_teacher_labels_but_execute_learner_and_preserve_rng():
    config = EnvConfig(grid_size=8, horizon=6)
    data, audit = collect_rollouts(config, [0], Park())
    env = WildfireFleetEnv(config)
    obs, _ = env.reset(seed=0)
    for i, row in enumerate(audit):
        maps, vector = tensor_observations(obs)
        torch.testing.assert_close(data['maps'][i], maps[0], rtol=0, atol=0)
        torch.testing.assert_close(data['vector'][i], vector[0], rtol=0, atol=0)
        assert row['executed_action'] == 0
        assert row['teacher_action'] == data['actions'][i].item()
        obs, *_ = env.step(0)
    assert any(row['teacher_action'] != row['executed_action'] for row in audit)
    assert (data['positions'] == data['base'][:, None]).all()
    teacher_data, _ = collect_rollouts(config, [0])
    assert data_digest(teacher_data) == data_digest(collect_teacher_data(config, [0]))


def test_merge_rejects_environment_mismatch_and_retains_scenario_identity():
    config = EnvConfig(grid_size=8, horizon=6)
    a, b = (collect_teacher_data(config, [seed]) for seed in [0, 1])
    merged = merge_data(a, b)
    assert len(merged['actions']) == 12
    assert set(merged['scenario_seeds'].tolist()) == {0, 1}
    bad = collect_teacher_data(replace(config, battery_capacity=90), [1])
    with pytest.raises(ValueError, match='environment'):
        merge_data(a, bad)


def test_matched_arms_have_identical_first_round_and_distinct_supervised_family(tmp_path):
    config = EnvConfig(grid_size=8, horizon=6)
    initial = collect_teacher_data(config, [0])
    validation = collect_teacher_data(config, [5000])
    settings = CoverageConfig(rounds=2, updates_per_round=4, batch_size=4)
    first_states = []
    for arm in ['fixed', 'teacher', 'dagger']:
        out = tmp_path/arm
        result = fit_coverage(config, initial, validation, [[1]], settings, 0, arm, out)
        assert result['completed_updates'] == 8
        first_states.append(load_coverage(out/'round-1.pt').model.state_dict())
        with pytest.raises(ValueError, match='imitation probe'):
            load_probe(out/'final_coverage.pt')
        with pytest.raises(KeyError):
            load_policy(out/'final_coverage.pt')
        assert load_coverage(out/'final_coverage.pt').env_config == config
        manifest = json.loads((out/'checkpoint_manifest.json').read_text())
        assert manifest['round-2.pt']['completed_updates'] == 8
    assert all(torch.equal(first_states[0][k], state[k]) for state in first_states[1:]
               for k in first_states[0])


def test_validation_labels_never_change_final_weights_or_collected_learner_data(tmp_path):
    config = EnvConfig(grid_size=8, horizon=6)
    initial = collect_teacher_data(config, [0])
    val = collect_teacher_data(config, [5000])
    altered = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in val.items()}
    altered['actions'] = (altered['actions']+1) % 125
    settings = CoverageConfig(rounds=2, updates_per_round=4, batch_size=4)
    for data, out in [(val, tmp_path/'a'), (altered, tmp_path/'b')]:
        fit_coverage(config, initial, data, [[1]], settings, 0, 'dagger', out)
    a, b = (load_coverage(tmp_path/p/'final_coverage.pt').model.state_dict() for p in ['a', 'b'])
    assert all(torch.equal(a[k], b[k]) for k in a)
    datasets = [torch.load(tmp_path/p/'added-round-2.pt', weights_only=True) for p in ['a', 'b']]
    assert data_digest(datasets[0]) == data_digest(datasets[1])


def test_coverage_preflights_overlap_and_preserves_existing_artifacts(tmp_path):
    config = EnvConfig(grid_size=8, horizon=6)
    initial = collect_teacher_data(config, [0])
    validation = collect_teacher_data(config, [5000])
    settings = CoverageConfig(rounds=2, updates_per_round=4)
    with pytest.raises(ValueError, match='overlap'):
        fit_coverage(config, initial, validation, [[5000]], settings, 0, 'dagger', tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
    out = tmp_path/'existing'
    out.mkdir()
    (out/'sentinel').write_text('preserve')
    with pytest.raises(ValueError, match='existing'):
        fit_coverage(config, initial, validation, [[1]], settings, 0, 'dagger', out)
    assert (out/'sentinel').read_text() == 'preserve'
    with pytest.raises(ValueError):
        CoverageConfig(updates_per_round=0)


def test_runner_binds_missions_to_checkpoints_and_rejects_overwrites_before_collection(tmp_path,
                                                                                    monkeypatch):
    import csv
    import hashlib
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1]/'scripts/run_coverage_probe.py'
    spec = importlib.util.spec_from_file_location('coverage_runner', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config = EnvConfig(grid_size=8, horizon=6)
    out = tmp_path/'run'
    protocol = runner.run_study(config, [0], [[1]], [5000], [0],
                                CoverageConfig(rounds=2, updates_per_round=4, batch_size=4), out)
    assert hashlib.sha256((out/'initial_teacher_data.pt').read_bytes()).hexdigest() == \
        protocol['initial_data']['sha256']
    rows = list(csv.DictReader((out/'validation_missions.csv').open()))
    assert len(rows) == 10
    assert rows[-1]['policy'] == 'greedy'
    for row in rows[:-1]:
        directory = out/row['arm']/'seed-0'
        record = json.loads((directory/'checkpoint_manifest.json').read_text())[row['checkpoint']]
        assert row['checkpoint_sha256'] == record['sha256']
        assert int(row['completed_updates']) == record['completed_updates']
    checkpoint = out/'dagger/seed-0/final_coverage.pt'
    record = json.loads((checkpoint.parent/'checkpoint_manifest.json').read_text())[checkpoint.name]
    assert 'completed_epochs' not in torch.load(checkpoint, weights_only=True)
    with pytest.raises(ValueError, match='bytes differ'):
        load_coverage(checkpoint, record | {'sha256': 'bad'})

    def forbidden(*args):
        pytest.fail('preflight should reject before expensive teacher collection')

    monkeypatch.setattr(runner, 'collect_rollouts', forbidden)
    with pytest.raises(ValueError, match='existing'):
        runner.run_study(config, [0], [[1]], [5000], [0],
                         CoverageConfig(rounds=2), out)
    with pytest.raises(ValueError, match='overlap'):
        runner.run_study(config, [0], [[5000]], [5000], [0],
                         CoverageConfig(rounds=2), tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
