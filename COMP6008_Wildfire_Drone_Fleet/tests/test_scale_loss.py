import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


def runner(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / 'scripts'
    path = directory / 'run_scale_loss.py'
    assert path.is_file(), 'controlled loss runner is missing'
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('scale_loss_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loss_counts_all_tied_best_actions_without_penalizing_choice(monkeypatch):
    loss = runner(monkeypatch).ranking_loss
    scores = torch.tensor([[2., 0., -1., 100.]], requires_grad=True)
    gains = torch.tensor([[3., 3., 1., 999.]], dtype=torch.float64)
    mask = torch.tensor([[True, True, True, False]])
    actual = loss(scores, gains, mask)
    expected = -torch.log(torch.softmax(scores[0, :3], 0)[:2].sum())
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert scores.grad[0, 3] == 0
    assert scores.grad[0, 0] < 0 and scores.grad[0, 1] < 0 and scores.grad[0, 2] > 0
    # Probability on either equivalent optimum is sufficient; no uniform-target penalty.
    concentrated = torch.tensor([[40., -40., -40., 100.]])
    assert float(loss(concentrated, gains, mask)) < 1e-6


def test_loss_ignores_infeasible_logits_and_is_stable_for_ties(monkeypatch):
    loss = runner(monkeypatch).ranking_loss
    scores = torch.tensor([[10000., 9999., -10000.]], requires_grad=True)
    gains = torch.tensor([[1., 1. + 5e-13, 9.]], dtype=torch.float64)
    mask = torch.tensor([[True, True, False]])
    value = loss(scores, gains, mask)
    assert float(value) == 0
    value.backward()
    assert torch.isfinite(scores.grad).all() and torch.count_nonzero(scores.grad) == 0
    with pytest.raises(ValueError):
        loss(scores, gains, torch.zeros_like(mask))
    with pytest.raises(ValueError):
        loss(torch.tensor([[float('nan'), 0., 0.]]), gains, mask)


def test_fit_matches_parent_initialization_and_sampling(tmp_path, monkeypatch):
    from wild_drone.centered_inspection import fit_centered, load_centered_model
    from wild_drone.config import EnvConfig
    from wild_drone.inspection_targets import collect_states

    module = runner(monkeypatch)
    config = EnvConfig(grid_size=8, horizon=8)
    data, _, _ = collect_states(config, (0, 1), (2, 4))
    _, control, _, initial = fit_centered(config, data, arm='centered', seed=0,
                                          updates=3, batch_size=2, output=tmp_path/'control')
    model, identity, history = module.fit_ranking(
        config, data, seed=0, updates=3, batch_size=2, output=tmp_path/'ranking',
        parent_initial=tmp_path/'control/initial.pt',
        expected_sequence=control['sample_sequence_sha256'])
    assert identity['sample_sequence_sha256'] == control['sample_sequence_sha256']
    assert identity['initial_tensors_match_parent']
    assert len(history) == 3 and np.isfinite([r['training_batch_loss'] for r in history]).all()
    saved = torch.load(tmp_path/'ranking/initial.pt', weights_only=True)
    assert all(torch.equal(v, saved['state_dict'][k]) for k, v in initial.items())
    loaded = load_centered_model(tmp_path/'ranking/final.pt', config, 'centered')
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())
    with pytest.raises(ValueError, match='sampling'):
        module.fit_ranking(config, data, seed=0, updates=3, batch_size=2,
                           output=tmp_path/'wrong', parent_initial=tmp_path/'control/initial.pt',
                           expected_sequence='wrong')


def test_effect_requires_each_seed_and_preserves_original_gates(monkeypatch):
    module = runner(monkeypatch)
    ranking, parent_ranking, episodes = [], [], []
    for seed in range(3):
        for split, regret in [('training', .02), ('fresh', .08)]:
            ranking.append(dict(seed=seed, phase='final', policy='network', split=split,
                                normalised_gain_regret=regret, one_step_optimal=.5))
            parent_ranking.append(dict(seed=str(seed), phase='final', policy='network',
                                       split=split, normalised_gain_regret=regret+.02))
        for name, freshness in [('ranking', .28), ('centered', .25)]:
            episodes += [dict(policy=name, training_seed=seed, scenario_seed=w,
                              weighted_freshness=freshness, overdue_fraction=.52,
                              total_losses=0, fleet_failure=0, **{'return': -4})
                         for w in (7000, 7001)]
    for name, fresh, overdue, reward in [('greedy', .3, .5, -3), ('masked_random', .1, .8, -12)]:
        episodes += [dict(policy=name, training_seed=None, scenario_seed=w,
                          weighted_freshness=fresh, overdue_fraction=overdue,
                          total_losses=0, fleet_failure=0, **{'return': reward})
                     for w in (7000, 7001)]
    result = module.loss_decisions(ranking, parent_ranking, episodes)
    assert all(r['effect_passed'] and r['mission_passed'] for r in result)
    next(r for r in episodes if r['policy']=='ranking' and r['training_seed']==2)[
        'weighted_freshness'] = .20
    result = module.loss_decisions(ranking, parent_ranking, episodes)
    assert result[0]['effect_passed'] and not result[2]['effect_passed']
    with pytest.raises(ValueError):
        module.loss_decisions(ranking, parent_ranking, episodes[:-1])


def auditor(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / 'scripts'
    path = directory / 'audit_scale_loss.py'
    assert path.is_file(), 'independent loss audit is missing'
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('scale_loss_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loss_audit_replays_real_mission_and_rejects_action_drift(monkeypatch):
    import copy

    from wild_drone.centered_inspection import CenteredInspectionNetwork
    from wild_drone.config import EnvConfig

    audit = auditor(monkeypatch)
    from run_mission_validation import ScorePolicy, rollout

    config = EnvConfig(grid_size=8, horizon=4)
    model = CenteredInspectionNetwork(config, 'centered').eval()
    episode, trace, vectors = rollout(config, ScorePolicy(config, model=model), 7000,
                                      name='ranking', training_seed=0)
    def strings(row):
        return {k: '' if v is None else str(v) for k, v in row.items()}

    episode, trace = strings(episode), [strings(r) for r in trace]
    verified, drift = audit.audit_world(config, model, episode, trace,
                                        vectors['scores'], vectors['masks'])
    assert verified['steps'] == 4 and drift < 1e-5
    bad = copy.deepcopy(trace)
    bad[0]['action'] = str((int(bad[0]['action'])+1) % 125)
    with pytest.raises(AssertionError):
        audit.audit_world(config, model, episode, bad, vectors['scores'], vectors['masks'])
    corrupt = vectors['scores'].copy()
    corrupt[0, 0] += 1
    with pytest.raises(AssertionError):
        audit.audit_world(config, model, episode, trace, corrupt, vectors['masks'])


def test_loss_audit_rejects_removed_summary_fields(monkeypatch):
    audit = auditor(monkeypatch)
    expected = {'effect_passed': False, 'per_seed': [{'seed': 0, 'freshness': .2}]}
    audit.equal_tree(expected, expected)
    with pytest.raises(AssertionError):
        audit.equal_tree({'effect_passed': False}, expected)
    with pytest.raises(AssertionError):
        audit.equal_tree({'effect_passed': True, 'per_seed': expected['per_seed']}, expected)
    with pytest.raises(AssertionError):
        audit.equal_tree({'effect_passed': 0, 'per_seed': expected['per_seed']}, expected)


def test_aggregate_uses_seed_means_instead_of_pooled_missions(monkeypatch):
    module = runner(monkeypatch)
    rows = []
    for seed, values in enumerate(((1., 3.), (3., 5.), (5., 7.))):
        for world, value in enumerate(values):
            rows.append({'policy': 'ranking', 'training_seed': seed, 'scenario_seed': world,
                         **{k: value for k in module.METRICS}})
    rows += [{'policy': 'greedy', 'training_seed': None, 'scenario_seed': w,
              **{k: value for k in module.METRICS}} for w, value in enumerate((2., 4.))]
    result = module.aggregate(rows)
    rank = next(r for r in result if r['policy']=='ranking' and r['metric']=='weighted_freshness')
    assert rank['unit'] == 'fitted_seed_means' and rank['units'] == 3
    assert rank['mean'] == 4. and rank['sample_sd'] == 2.
    baseline = next(r for r in result if r['policy']=='greedy' and
                    r['metric']=='weighted_freshness')
    assert baseline['unit'] == 'worlds' and baseline['units'] == 2
    assert baseline['sample_sd'] == pytest.approx(np.sqrt(2))


def test_ranking_audit_accepts_real_export_metadata(tmp_path, monkeypatch):
    from wild_drone.centered_inspection import fit_centered, load_centered_model
    from wild_drone.config import EnvConfig
    from wild_drone.experiments import write_rows
    from wild_drone.inspection_targets import collect_states, score_model

    audit = auditor(monkeypatch)
    from run_centered_inspection import metrics, summary

    config = EnvConfig(grid_size=8, horizon=8)
    packet, records, _ = collect_states(config, (0, 1), (2, 4))
    data = {'training': packet, 'fresh': packet}
    index = {split: [r | {'gains': packet['gains'][i].tolist(),
                         'mask': packet['masks'][i].tolist()} for i, r in enumerate(records)]
             for split in data}
    ranks = tmp_path/'rankings'
    ranks.mkdir()
    natural, means = [], []
    for seed in range(3):
        fit_centered(config, packet, arm='centered', seed=seed, updates=2, batch_size=2,
                     output=tmp_path/f'seed-{seed}')
        for phase in ('initial', 'final'):
            model = load_centered_model(tmp_path/f'seed-{seed}/{phase}.pt', config, 'centered')
            blocks = {}
            for split in (('fresh',) if phase=='initial' else ('training', 'fresh')):
                blocks[split] = score_model(model, packet)
                rows = metrics(blocks[split], packet, index[split], split=split, seed=seed,
                               phase=phase, policy='network')
                natural.extend(rows)
                means.append({'split': split, 'seed': seed, 'phase': phase,
                              'policy': 'network', **summary(rows)})
            np.savez_compressed(ranks/f'seed-{seed}-{phase}.npz', **blocks)
    write_rows(ranks/'natural.csv', natural)
    write_rows(ranks/'per-fit.csv', means)
    verified, count, drift = audit.audit_rankings(tmp_path, config, index, data)
    assert count == len(natural) and len(verified)==9 and drift < 1e-5
