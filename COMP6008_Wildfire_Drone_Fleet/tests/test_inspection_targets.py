import copy

import numpy as np
import pytest
import torch

from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv


def probe():
    from wild_drone import inspection_targets
    return inspection_targets


def test_targets_preserve_feasible_orders_and_ties_without_zero_range_division():
    gains = np.zeros((2, 125))
    masks = np.zeros((2, 125), dtype=bool)
    gains[0, :4] = [2, 4, 4, 1000]
    masks[:, :3] = True
    targets, informative = probe().normalised_targets(gains, masks)
    np.testing.assert_array_equal(targets[0, :4], [0, 1, 1, 0])
    np.testing.assert_array_equal(targets[1], 0)
    np.testing.assert_array_equal(informative, [True, False])


def test_loss_ignores_infeasible_outputs_and_weights_states_equally():
    scores = torch.zeros(2, 125, requires_grad=True)
    targets = torch.zeros_like(scores)
    targets[0, :2], targets[1, :4] = 1, 2
    masks = torch.zeros_like(scores, dtype=torch.bool)
    masks[0, :2], masks[1, :4] = True, True
    loss = probe().masked_target_loss(scores, targets, masks)
    assert float(loss.detach()) == 2.5
    loss.backward()
    assert torch.count_nonzero(scores.grad[~masks]) == 0


def test_invalid_targets_and_overlapping_worlds_rejected():
    for masks in [np.zeros((1, 125), dtype=bool), np.ones((1, 124), dtype=bool)]:
        with pytest.raises(ValueError):
            probe().normalised_targets(np.zeros((1, 125)), masks)
    with pytest.raises(ValueError):
        probe().normalised_targets(np.full((1, 125), np.nan),
                                   np.ones((1, 125), dtype=bool))
    with pytest.raises(ValueError, match='overlap'):
        probe().validate_worlds([1, 2], [2, 3])
    with pytest.raises(ValueError):
        probe().validate_worlds([1, 1], [3])


def test_collection_records_actual_states_targets_and_does_not_train_on_counterfactuals():
    config = EnvConfig(grid_size=8, horizon=8)
    data, records, missing = probe().collect_states(config, [11], [2, 4])
    assert len(records) == 4 and not missing
    assert set(data) == {'maps', 'vectors', 'gains', 'masks', 'targets', 'informative'}
    env = WildfireFleetEnv(config)
    obs, _ = env.reset(seed=11)
    from wild_drone.masked_learning import MaskedRandomPolicy, action_mask
    controller = MaskedRandomPolicy(11)
    for _ in range(2):
        obs = env.step(controller.act(env, obs))[0]
    before = copy.deepcopy(env)
    from wild_drone.ranking_probe import gain_scores, transition_coverage
    np.testing.assert_allclose(data['gains'][0], gain_scores(env, transition_coverage(env)))
    np.testing.assert_array_equal(data['masks'][0], action_mask(env))
    np.testing.assert_array_equal(env.ages, before.ages)
    assert records[0]['case'] == '11-masked_random-2'


def test_fitting_export_roundtrip_and_rl_loader_rejection(tmp_path):
    config = EnvConfig(grid_size=8, horizon=8)
    data, _, _ = probe().collect_states(config, [12], [2])
    model, manifest, history, initial = probe().fit_targets(
        config, data, seed=3, updates=2, batch_size=2, output=tmp_path)
    assert len(history) == 2 and manifest['completed_updates'] == 2
    loaded = probe().load_target_model(tmp_path/'final.pt', config)
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())
    assert any(not torch.equal(v, initial[k]) for k, v in model.state_dict().items())
    with pytest.raises(ValueError):
        probe().load_target_model(tmp_path/'final.pt', EnvConfig(grid_size=9))
    from wild_drone.learning import load_policy
    from wild_drone.masked_learning import load_masked_policy
    with pytest.raises(KeyError):
        load_policy(tmp_path/'final.pt')
    with pytest.raises(ValueError):
        load_masked_policy(tmp_path/'final.pt')


def test_fixed_fit_reproduces_initial_and_final_tensors(tmp_path):
    config = EnvConfig(grid_size=8, horizon=8)
    data, _, _ = probe().collect_states(config, [13], [2, 4])
    a, _, ah, ai = probe().fit_targets(
        config, data, seed=4, updates=3, batch_size=2, output=tmp_path/'a')
    b, _, bh, bi = probe().fit_targets(
        config, data, seed=4, updates=3, batch_size=2, output=tmp_path/'b')
    assert ah == bh
    assert all(torch.equal(v, bi[k]) for k, v in ai.items())
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def test_runner_end_to_end_with_small_test_only_budget(monkeypatch, tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1]/'scripts'/'run_inspection_targets.py'
    spec = importlib.util.spec_from_file_location('target_runner', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, 'TRAIN_WORLDS', (11,))
    monkeypatch.setattr(runner, 'HELDOUT_WORLDS', (12,))
    monkeypatch.setattr(runner, 'STEPS', (2,))
    fit = runner.fit_targets
    monkeypatch.setattr(runner, 'fit_targets', lambda config, data, **kw: fit(
        config, data, **(kw | {'updates': 2})))

    def pairs(config):
        data, records, _ = runner.collect_states(config, [12], [2])
        records = [r | {'case': r['case']+':natural'} for r in records]
        return data, records, [{'case': '12-masked_random-2', 'kind': 'noop',
                                'before': 0, 'after': 0}], []

    monkeypatch.setattr(runner, 'collect_pairs', pairs)
    out = tmp_path/'test-only-run'
    runner.main(['--out', str(out), '--source-revision', 'a'*40])
    report = json.loads((out/'completion.json').read_text())
    assert report['observed_states'] == {'training': 2, 'heldout': 2, 'probes': 2}
    assert report['probe_pair_counts']['noop'] == 1
    assert len(json.loads((out/'checkpoints.json').read_text())) == 3
    assert (out/'natural.csv').exists() and (out/'pairs.csv').exists()
