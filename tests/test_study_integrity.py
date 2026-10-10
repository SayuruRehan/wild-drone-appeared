import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from wild_drone.config import EnvConfig
from wild_drone.learning import TrainConfig, train
from wild_drone.masked_learning import train_masked

ROOT = Path(__file__).parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONFIG = EnvConfig(grid_size=8, horizon=4)
SETTINGS = TrainConfig(episodes=1, batch_size=4, warmup_steps=4,
                       validation_seeds=(5000,), progress_interval=0)


@pytest.mark.parametrize('masked', [False, True])
def test_finished_fit_can_be_reused_only_for_the_same_request(masked, tmp_path):
    runner = script('run_final_study')
    runner.fit((masked, 'dqn', CONFIG, SETTINGS, tmp_path))
    before = (tmp_path/'policy.pt').read_bytes()
    assert runner.fit((masked, 'dqn', CONFIG, SETTINGS, tmp_path)) == str(tmp_path)
    assert (tmp_path/'policy.pt').read_bytes() == before
    for config, settings in [(CONFIG, replace(SETTINGS, episodes=2)),
                             (CONFIG, replace(SETTINGS, learning_rate=0.001)),
                             (replace(CONFIG, battery_capacity=100), SETTINGS)]:
        with pytest.raises(ValueError, match='cached'):
            runner.fit((masked, 'dqn', config, settings, tmp_path))
    assert (tmp_path/'policy.pt').read_bytes() == before


@pytest.mark.parametrize('damage', ['missing_policy', 'wrong_mask', 'wrong_checkpoint',
                                  'short_training', 'short_validation', 'missing_metrics',
                                  'nonfinite_metric'])
def test_cache_requires_complete_matching_exports(damage, tmp_path):
    runner = script('run_final_study')
    runner.fit((True, 'dqn', CONFIG, SETTINGS, tmp_path))
    checkpoint = tmp_path/'policy.pt'
    if damage == 'missing_policy':
        checkpoint.unlink()
    elif damage in ('wrong_mask', 'wrong_checkpoint'):
        data = torch.load(checkpoint, weights_only=True)
        if damage == 'wrong_mask':
            data.pop('action_mask')
        else:
            data['completed_episodes'] = 0
        torch.save(data, checkpoint)
    elif damage in ('short_training', 'short_validation'):
        csv = tmp_path/('training.csv' if damage == 'short_training' else 'validation.csv')
        csv.write_text(csv.read_text().splitlines()[0]+'\n')
    elif damage == 'missing_metrics':
        (tmp_path/'training.csv').write_text(
            'episode,scenario_seed,training_seed\n1,0,0\n')
    else:
        import csv

        path = tmp_path/'validation.csv'
        rows = list(csv.DictReader(path.open()))
        rows[0]['return'] = 'nan'
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with pytest.raises(ValueError, match='cached'):
        runner.fit((True, 'dqn', CONFIG, SETTINGS, tmp_path))


def test_masked_study_uses_the_same_cache_guard(tmp_path):
    runner = script('run_masked_study')
    runner.fit(('dqn', CONFIG, SETTINGS, tmp_path))
    with pytest.raises(ValueError, match='cached'):
        runner.fit(('dqn', CONFIG, replace(SETTINGS, episodes=2), tmp_path))


@pytest.mark.parametrize('stage', ['tune', 'test', 'sensitivity'])
def test_study_rejects_training_test_overlap_before_creating_runs(stage, tmp_path):
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'environment': CONFIG.to_dict(),
                                 'training': {'scenario_seed_start': 10000}}))
    out = tmp_path/'out'
    with pytest.raises(ValueError, match='overlap'):
        script('run_final_study').main([stage, '--config', str(config), '--episodes', '1',
                                       '--seeds', '0', '--jobs', '1', '--out', str(out),
                                       '--settings', str(ROOT/'configs/frozen-settings.json')])
    assert not out.exists()


def test_all_cached_jobs_are_checked_before_any_training(tmp_path):
    runner = script('run_final_study')
    bad = tmp_path/'bad'
    train_masked('dqn', CONFIG, SETTINGS, bad)
    fresh = tmp_path/'fresh'
    with pytest.raises(ValueError, match='cached'):
        runner.run_jobs([(True, 'dqn', CONFIG, SETTINGS, fresh),
                         (True, 'dqn', CONFIG, replace(SETTINGS, seed=1), bad)], 1)
    assert not fresh.exists()


def test_unknown_mask_is_not_silently_loaded_as_unmasked(tmp_path):
    from wild_drone.learning import load_policy

    train('dqn', CONFIG, SETTINGS, tmp_path)
    data = torch.load(tmp_path/'policy.pt', weights_only=True)
    data['action_mask'] = 'unrecognised-rule'
    torch.save(data, tmp_path/'policy.pt')
    with pytest.raises(ValueError, match='mask'):
        load_policy(tmp_path/'policy.pt')


def test_old_double_dqn_metadata_cannot_be_reused_as_a_corrected_fit(tmp_path):
    from wild_drone.learning import load_policy

    settings = replace(SETTINGS, double_dqn=True)
    train_masked('dqn', CONFIG, settings, tmp_path)
    data = torch.load(tmp_path/'policy.pt', weights_only=True)
    data.pop('dqn_target_rule', None)
    torch.save(data, tmp_path/'policy.pt')
    with pytest.raises(ValueError, match='target rule'):
        load_policy(tmp_path/'policy.pt')
    with pytest.raises(ValueError, match='cached'):
        script('run_final_study').fit((True, 'dqn', CONFIG, settings, tmp_path))
