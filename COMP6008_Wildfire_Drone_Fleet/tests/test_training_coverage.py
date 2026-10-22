import importlib.util
from pathlib import Path

import pytest
import torch

from wild_drone.config import EnvConfig
from wild_drone.inspection_targets import collect_states


def runner(monkeypatch):
    directory = Path(__file__).resolve().parents[1]/'scripts'
    path = directory/'run_training_coverage.py'
    assert path.is_file(), 'training-coverage runner has not been implemented'
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('coverage_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recipe_requires_nested_training_and_separate_evaluation_worlds(monkeypatch):
    module = runner(monkeypatch)
    module.validate_recipe((0, 1), (0, 1, 2, 3), (20, 21), (30, 31))
    for expanded, fresh, reused in [((0, 2, 3), (20,), (30,)),
                                    ((0, 1), (20,), (30,)),
                                    ((0, 1, 2), (2,), (30,)),
                                    ((0, 1, 2), (20,), (20,)),
                                    ((0, 1, 1, 2), (20,), (30,))]:
        with pytest.raises(ValueError):
            module.validate_recipe((0, 1), expanded, fresh, reused)


def test_extension_preserves_original_states_and_rejects_duplicate_worlds(monkeypatch):
    module = runner(monkeypatch)
    config = EnvConfig(grid_size=8, horizon=10)
    small, records, _ = collect_states(config, [21], [2, 4])
    more, other, _ = collect_states(config, [22], [2, 4])
    expanded, index = module.extend_data(small, more, records, other)
    assert index == records+other
    assert len(expanded['maps']) == len(small['maps'])+len(more['maps'])
    for key, tensor in small.items():
        assert torch.equal(expanded[key][:len(tensor)], tensor)
        assert torch.equal(expanded[key][len(tensor):], more[key])
    expanded['maps'][0] = 0
    assert small['maps'][0].any()
    with pytest.raises(ValueError):
        module.extend_data(small, small, records, records)


def test_every_seed_must_pass_separate_effect_and_capability_conditions(monkeypatch):
    module = runner(monkeypatch)
    rows = []
    for seed in (0, 1, 2):
        for corpus, split, regret, best in [('small', 'fresh', .4, .1),
                                           ('expanded', 'fresh', .15, .3),
                                           ('expanded', 'training', .02, .9)]:
            rows.append(dict(seed=seed, corpus=corpus, split=split, phase='final',
                             policy='network', normalised_gain_regret=regret,
                             one_step_optimal=best))
    result = module.criteria(rows)
    assert all(r['data_effect_passed'] and r['capability_passed'] for r in result)
    rows[-2]['normalised_gain_regret'] = .39
    rows[-2]['one_step_optimal'] = .08
    result = module.criteria(rows)
    assert result[0]['data_effect_passed'] and result[1]['capability_passed']
    assert not result[2]['data_effect_passed'] and not result[2]['capability_passed']


def test_end_to_end_exports_reproduce_both_corpora_and_shared_initial_scores(
        tmp_path, monkeypatch):
    import json

    import numpy as np

    from wild_drone.centered_inspection import fit_centered, load_centered_model
    from wild_drone.experiments import write_rows
    from wild_drone.inspection_targets import score_model

    module = runner(monkeypatch)
    config = EnvConfig(grid_size=8, horizon=10)
    small, records, _ = collect_states(config, [21], [2, 4])
    reused, previous, _ = collect_states(config, [23], [2, 4])
    monkeypatch.setattr(module, 'load_prior', lambda *_: (config, small, reused, records, previous))
    monkeypatch.setattr(module, 'SMALL_WORLDS', (21,))
    monkeypatch.setattr(module, 'EXPANDED_WORLDS', (21, 22))
    monkeypatch.setattr(module, 'FRESH_WORLDS', (24,))
    monkeypatch.setattr(module, 'REUSED_WORLDS', (23,))
    monkeypatch.setattr(module, 'STEPS', (2, 4))
    monkeypatch.setattr(module, 'UPDATES', 2)
    monkeypatch.setattr(module, 'BATCH_SIZE', 2)
    parent, output = tmp_path/'parent', tmp_path/'output'
    for seed in (0, 1, 2):
        _, _, rows, _ = fit_centered(config, small, arm='centered', seed=seed, updates=2,
                                    batch_size=2, output=parent/'centered'/f'seed-{seed}')
        write_rows(parent/'centered'/f'seed-{seed}'/'training.csv', rows)
    module.main(['--parent-root', str(tmp_path/'unused'), '--centered-root', str(parent),
                 '--out', str(output), '--source-revision', 'a'*40])
    matching = json.loads((output/'matching.json').read_text())
    assert len(matching) == 3
    assert all(r['small_final_and_training_history_match_parent'] for r in matching)
    datasets = {k: torch.load(output/f'{k}.pt', weights_only=True)
                for k in ('small', 'expanded', 'fresh', 'reused')}
    for seed in (0, 1, 2):
        for corpus in ('small', 'expanded'):
            directory = output/corpus/f'seed-{seed}'
            model = load_centered_model(directory/'final.pt', config, 'centered')
            with np.load(directory/'final-scores.npz', allow_pickle=False) as scores:
                for split in ('training', 'fresh', 'reused'):
                    data = datasets[corpus if split == 'training' else split]
                    np.testing.assert_array_equal(scores[split], score_model(model, data))
        model = load_centered_model(output/'small'/f'seed-{seed}'/'initial.pt', config, 'centered')
        with np.load(output/'initial'/f'seed-{seed}-scores.npz', allow_pickle=False) as scores:
            np.testing.assert_array_equal(scores['fresh'], score_model(model, datasets['fresh']))
    with pytest.raises(FileExistsError):
        module.main(['--parent-root', str(tmp_path/'unused'), '--centered-root', str(parent),
                     '--out', str(output), '--source-revision', 'a'*40])
