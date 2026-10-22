import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def module(monkeypatch, name):
    directory = Path(__file__).resolve().parents[1] / 'scripts'
    path = directory / f'{name}.py'
    assert path.is_file(), f'{name} is missing'
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_frozen_rl_recipe_is_nine_fits_without_selection(monkeypatch):
    run = module(monkeypatch, 'run_scale_rl')
    config, settings = run.recipe()
    assert config.grid_size == 40 and config.horizon == 300
    assert set(settings) == {'dqn', 'reinforce', 'a2c'}
    for algorithm in settings:
        for seed in run.SEEDS:
            t = run.training_config(settings, algorithm, seed)
            assert t.episodes == 300 and t.seed == seed
            assert t.validation_seeds == () and t.coordination_shaping_weight == 0
            assert not t.double_dqn and t.device == 'cpu'
    assert len(run.ALGORITHMS)*len(run.SEEDS) == 9
    assert set(run.WORLDS) == set(range(22000, 22020))
    assert not set(range(300)) & set(run.WORLDS)


def test_fresh_world_check_reads_actual_scenario_columns(tmp_path, monkeypatch):
    run = module(monkeypatch, 'run_scale_rl')
    (tmp_path/'old.csv').write_text('scenario_seed,return\n7000,22000.0\n')
    run.check_fresh_worlds(tmp_path, (22000,))
    (tmp_path/'used.csv').write_text('scenario_seed,return\n22000,-1\n')
    with pytest.raises(ValueError, match='already'):
        run.check_fresh_worlds(tmp_path, (22000,))


def test_negative_archive_check_rejects_changed_evidence(tmp_path, monkeypatch):
    run = module(monkeypatch, 'run_scale_rl')
    archive = tmp_path/'previous'
    archive.mkdir()
    evidence = archive/'negative.csv'
    evidence.write_text('scenario_seed,return\n10000,-2\n')
    (archive/'checksums.json').write_text(json.dumps({
        evidence.name: {'sha256': run.digest(evidence), 'bytes': evidence.stat().st_size}}))
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'NEGATIVE', ('previous/checksums.json',))
    run.verify_negative_archives()
    evidence.write_text('scenario_seed,return\n10000,-1\n')
    with pytest.raises(AssertionError):
        run.verify_negative_archives()


def string_rows(row):
    return {k: '' if v is None else str(v) for k, v in row.items()}


def test_true_rl_scores_replay_and_reject_drift(monkeypatch):
    from wild_drone.config import EnvConfig
    from wild_drone.learning import FleetNetwork
    from wild_drone.masked_learning import MaskedNeuralPolicy

    run, audit = (module(monkeypatch, name) for name in ('run_scale_rl', 'audit_scale_rl'))
    config = EnvConfig(grid_size=8, horizon=4)
    model = FleetNetwork(8, critic=True, spatial_encoding=True).eval()
    policy = run.RLScorePolicy(config, model=model)
    episode, trace, vectors = run.rollout(config, policy, 22000, name='a2c', training_seed=0)
    replay_policy = MaskedNeuralPolicy(model, 'a2c', config)
    actual, drift = audit.replay_world(config, replay_policy, string_rows(episode),
                                       [string_rows(r) for r in trace], vectors)
    assert actual['steps'] == 4 and drift == 0
    bad = copy.deepcopy(trace)
    bad[0]['action'] = (bad[0]['action']+1) % 125
    with pytest.raises(AssertionError):
        audit.replay_world(config, replay_policy, string_rows(episode),
                           [string_rows(r) for r in bad], vectors)
    bad_scores = {k: v.copy() for k, v in vectors.items()}
    bad_scores['scores'][0, 0] += 1
    with pytest.raises(AssertionError):
        audit.replay_world(config, replay_policy, string_rows(episode),
                           [string_rows(r) for r in trace], bad_scores)


def test_random_baseline_replay_accounts_for_early_fleet_loss(monkeypatch):
    from wild_drone.baselines import RandomPolicy
    from wild_drone.config import EnvConfig

    run, audit = (module(monkeypatch, name) for name in ('run_scale_rl', 'audit_scale_rl'))
    config = EnvConfig(grid_size=8, horizon=20, battery_capacity=2)
    episode, trace, vectors = run.rollout(config, RandomPolicy(22000), 22000, name='random')
    actual, _ = audit.replay_world(config, RandomPolicy(22000), string_rows(episode),
                                  [string_rows(r) for r in trace], vectors)
    assert vectors is None and actual['fleet_failure'] == 1 and actual['steps'] < 20
    assert actual['weighted_freshness'] == episode['weighted_freshness']
    assert actual['overdue_fraction'] == episode['overdue_fraction']
    with pytest.raises(AssertionError):
        audit.replay_world(config, RandomPolicy(3), string_rows(episode),
                           [string_rows(r) for r in trace], vectors)


def test_summaries_use_seed_means_and_require_all_fits(monkeypatch):
    run, audit = (module(monkeypatch, name) for name in ('run_scale_rl', 'audit_scale_rl'))
    rows = []
    for policy in run.ALGORITHMS + run.BASELINES:
        for seed in run.SEEDS if policy in run.ALGORITHMS else (None,):
            for world in (22000, 22001):
                row = {key: 0. for key in run.METRICS}
                row.update(policy=policy, training_seed=seed, scenario_seed=world)
                row['return'] = (seed or 0)*2+(world-22000)*10
                rows.append(row)
    comparison, summary, paired, decisions = run.tables(rows, (22000, 22001))
    result = audit.independent_tables([string_rows(r) for r in rows], (22000, 22001))
    audit.equal_tree(result, (comparison, summary, paired, decisions))
    returns = next(r for r in summary if r['policy']=='dqn' and r['metric']=='return')
    assert returns['units'] == 3 and returns['sample_sd'] == 2
    assert len(paired) == 9*2*4
    with pytest.raises(ValueError):
        run.tables(rows[:-1], (22000, 22001))


@pytest.mark.parametrize('algorithm', ['dqn', 'reinforce', 'a2c'])
def test_real_mini_training_audit_rejects_budget_and_identity_changes(
        tmp_path, monkeypatch, algorithm):
    from wild_drone.config import EnvConfig
    from wild_drone.learning import TrainConfig
    from wild_drone.masked_learning import train_masked

    audit = module(monkeypatch, 'audit_scale_rl')
    config = EnvConfig(grid_size=8, horizon=4)
    settings = TrainConfig(episodes=2, seed=1, batch_size=2, warmup_steps=2,
                           replay_size=8, progress_interval=0, n_steps=2)
    train_masked(algorithm, config, settings, tmp_path/'fit')
    record = audit.verify_fit(tmp_path/'fit', algorithm, 1, config, settings,
                              source_revision=None)
    assert record['episodes'] == 2 and record['steps'] == 8 and record['updates'] > 0
    saved = torch.load(tmp_path/'fit/policy.pt', weights_only=True)
    saved['completed_episodes'] = 1
    torch.save(saved, tmp_path/'fit/policy.pt')
    with pytest.raises(AssertionError):
        audit.verify_fit(tmp_path/'fit', algorithm, 1, config, settings, source_revision=None)


def test_independent_schema_check_rejects_bool_gate_coercion(monkeypatch):
    audit = module(monkeypatch, 'audit_scale_rl')
    with pytest.raises(AssertionError):
        audit.equal_tree({'passed': 1}, {'passed': True})
    with pytest.raises(AssertionError):
        audit.equal_tree({'passed': True, 'extra': 1}, {'passed': True})
    with pytest.raises(AssertionError):
        audit.equal_tree(np.nan, 0.)
