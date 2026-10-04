import csv
import importlib.util
import json
from pathlib import Path

import pytest

from wild_drone.config import EnvConfig
from wild_drone.coverage_probe import CoverageConfig, fit_coverage, load_coverage
from wild_drone.imitation_probe import collect_teacher_data


def test_fixed_corpus_can_train_multiple_rounds_without_unused_scenario_batches(tmp_path):
    cfg = EnvConfig(grid_size=8, horizon=6)
    train = collect_teacher_data(cfg, [0, 1])
    val = collect_teacher_data(cfg, [5000])
    result = fit_coverage(cfg, train, val, [], CoverageConfig(rounds=2, updates_per_round=4),
                          0, 'fixed', tmp_path)
    assert result['completed_updates'] == 8
    assert result['final_training_states'] == 12
    assert not list(tmp_path.glob('added-round-*.pt'))
    assert load_coverage(tmp_path/'round-1.pt').env_config == cfg
    with pytest.raises(ValueError, match='scenario batch'):
        fit_coverage(cfg, train, val, [], CoverageConfig(rounds=2), 0, 'teacher', tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_budget_runner_binds_nested_budgets_and_preserves_existing_outputs(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1]/'scripts/run_fitting_budget_probe.py'
    assert path.exists(), 'the bounded budget runner must exist'
    spec = importlib.util.spec_from_file_location('fitting_budget_runner', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    cfg = EnvConfig(grid_size=8, horizon=6)
    out = tmp_path/'run'
    runner.run_budget_study(cfg, [0, 1], [5000], [0], [4, 8], 4, out)
    rows = list(csv.DictReader((out/'validation_missions.csv').open()))
    assert len(rows) == 4  # Two budgets, separately joint-selected policy, one Greedy control.
    manifest = json.loads((out/'seed-0/checkpoint_manifest.json').read_text())
    for row in rows[:-1]:
        assert row['checkpoint_sha256'] == manifest[row['checkpoint']]['sha256']
    assert [int(row['completed_updates']) for row in rows[:2]] == [4, 8]
    assert rows[-1]['policy'] == 'greedy'

    def forbidden(*args):
        pytest.fail('reject preflight before teacher collection')

    monkeypatch.setattr(runner, 'collect_rollouts', forbidden)
    with pytest.raises(ValueError, match='existing'):
        runner.run_budget_study(cfg, [0], [5000], [0], [4, 8], 4, out)
    with pytest.raises(ValueError, match='overlap'):
        runner.run_budget_study(cfg, [0], [0], [0], [4, 8], 4, tmp_path/'overlap')
    with pytest.raises(ValueError, match='budget'):
        runner.run_budget_study(cfg, [0], [5000], [0], [4, 9], 4, tmp_path/'bad-budget')
    assert not (tmp_path/'bad-budget').exists()


def test_budget_checkpoints_are_exact_prefixes_of_longer_training(tmp_path):
    import torch
    cfg = EnvConfig(grid_size=8, horizon=6)
    train = collect_teacher_data(cfg, [0, 1])
    val = collect_teacher_data(cfg, [5000])
    for rounds, name in [(2, 'short'), (4, 'long')]:
        fit_coverage(cfg, train, val, [], CoverageConfig(rounds=rounds, updates_per_round=4),
                     0, 'fixed', tmp_path/name)
    short = load_coverage(tmp_path/'short/final_coverage.pt').model.state_dict()
    long = load_coverage(tmp_path/'long/round-2.pt').model.state_dict()
    assert all(torch.equal(short[k], long[k]) for k in short)
