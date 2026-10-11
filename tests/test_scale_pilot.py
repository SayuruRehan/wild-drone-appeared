import importlib.util
from pathlib import Path

import pytest
import torch


def runner(monkeypatch):
    directory = Path(__file__).resolve().parents[1]/'scripts'
    path = directory/'run_scale_pilot.py'
    assert path.is_file(), '40x40 pilot runner has not been implemented'
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('scale_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_recipe_and_three_disjoint_world_sets(monkeypatch):
    module = runner(monkeypatch)
    config = module.pilot_config()
    assert (config.grid_size, config.horizon, config.battery_capacity, config.recharge_rate,
            config.sensor_radius, config.inspection_deadline, config.max_fuel) == \
        (40, 300, 160, 8, 2, 30, 180)
    module.validate_recipe((0, 1), (6800,), (7000,), (30, 60), config)
    for ranking, missions, steps in [((0,), (7000,), (30,)),
                                     ((6800,), (6800,), (30,)),
                                     ((6800, 6800), (7000,), (30,)),
                                     ((6800,), (100001,), (30,)),
                                     ((5800,), (7000,), (30,)),
                                     ((6800,), (7000,), (300,)),
                                     ((6800,), (7000,), (60, 30))]:
        with pytest.raises(ValueError):
            module.validate_recipe((0, 1), ranking, missions, steps, config)


def test_capability_requires_ranking_and_missions_for_every_seed(monkeypatch):
    module = runner(monkeypatch)
    ranking = []
    missions = [dict(policy=name, training_seed=None, scenario_seed=w,
                     weighted_freshness=fresh, overdue_fraction=overdue,
                     total_losses=0, fleet_failure=0, **{'return': reward})
                for name, fresh, overdue, reward in [('greedy', .3, .5, -4),
                                                     ('masked_random', .1, .8, -12)]
                for w in (7000, 7001)]
    for seed in (0, 1, 2):
        ranking += [dict(seed=seed, phase='final', policy='network', split=split,
                         normalised_gain_regret=regret, one_step_optimal=.4)
                    for split, regret in [('training', .03), ('fresh', .1)]]
        missions += [dict(policy='centered', training_seed=seed, scenario_seed=w,
                          weighted_freshness=.28, overdue_fraction=.53,
                          total_losses=0, fleet_failure=0, **{'return': -5}) for w in (7000, 7001)]
    decisions = module.pilot_decisions(ranking, missions)
    assert all(r['ranking_passed'] and r['mission_passed'] for r in decisions)
    ranking[-1]['normalised_gain_regret'] = .4
    missions[-1]['total_losses'] = 3
    decisions = module.pilot_decisions(ranking, missions)
    assert decisions[0]['mission_passed'] and not decisions[2]['ranking_passed']
    assert not decisions[2]['mission_passed']
    with pytest.raises(ValueError, match='matched'):
        module.pilot_decisions(ranking, missions[:-1])


def test_miniature_pilot_exports_matching_scores_and_fixed_fits(tmp_path, monkeypatch):
    import csv
    import gzip
    import json

    import numpy as np

    from wild_drone.centered_inspection import load_centered_model
    from wild_drone.config import EnvConfig
    from wild_drone.inspection_targets import score_model

    module = runner(monkeypatch)
    config = EnvConfig(grid_size=8, horizon=8)
    monkeypatch.setattr(module, 'pilot_config', lambda: config)
    monkeypatch.setattr(module, 'TRAIN_WORLDS', (0, 1))
    monkeypatch.setattr(module, 'RANKING_WORLDS', (6800,))
    monkeypatch.setattr(module, 'MISSION_WORLDS', (7000,))
    monkeypatch.setattr(module, 'STEPS', (2, 4))
    monkeypatch.setattr(module, 'UPDATES', 2)
    monkeypatch.setattr(module, 'BATCH_SIZE', 2)
    output = tmp_path/'pilot'
    argv = ['--out', str(output), '--source-revision', 'a'*40]
    module.main(argv)
    completion = json.loads((output/'completion.json').read_text())
    assert completion['mission_episodes'] == 7 and completion['rl_updates'] == 0
    with gzip.open(output/'missions'/'steps.csv.gz', 'rt') as f:
        trace = list(csv.DictReader(f))
    assert len(trace) == 7*8
    for seed in (0, 1, 2):
        directory = output/f'seed-{seed}'
        model = load_centered_model(directory/'final.pt', config, 'centered')
        initial = torch.load(directory/'initial.pt', weights_only=True)
        final = torch.load(directory/'final.pt', weights_only=True)
        assert initial['completed_updates'] == 0 and final['completed_updates'] == 2
        assert any(not torch.equal(v, initial['state_dict'][k])
                   for k, v in final['state_dict'].items())
        with np.load(output/'rankings'/f'seed-{seed}-final.npz', allow_pickle=False) as values:
            for split in ('training', 'fresh'):
                data = torch.load(output/f'{split}.pt', weights_only=True)
                np.testing.assert_array_equal(values[split], score_model(model, data))
        with np.load(output/'missions'/f'centered-seed-{seed}-scores.npz',
                     allow_pickle=False) as values:
            chosen = np.where(values['masks'], values['scores'], -np.inf).argmax(1)
            group = [r for r in trace if r['policy'] == 'centered' and
                     r['training_seed'] == str(seed)]
            assert chosen.tolist() == [int(r['action']) for r in group]
    with pytest.raises(FileExistsError):
        module.main(argv)
