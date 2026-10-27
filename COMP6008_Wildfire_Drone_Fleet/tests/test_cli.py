import csv
import json

import numpy as np
import pytest
from PIL import Image

from wild_drone.cli import main
from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.experiments import validate_seed_split


def test_rgb_render_and_demo_file(tmp_path):
    env = WildfireFleetEnv(EnvConfig(grid_size=8, horizon=8), render_mode='rgb_array')
    env.reset(seed=6)
    rgb = env.render()
    assert rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8
    out = tmp_path/'demo'
    main(['demo', '--grid', '8', '--horizon', '8', '--out', str(out)])
    with Image.open(out/'demo.gif') as image:
        assert image.n_frames > 1
    assert json.loads((out/'metadata.json').read_text())['env_config']['grid_size'] == 8


def test_baseline_evaluation_writes_metrics(tmp_path):
    main(['evaluate', '--policy', 'greedy', '--grid', '8', '--horizon', '8',
          '--scenario-seeds', '10000', '10001', '--out', str(tmp_path)])
    with (tmp_path/'evaluation.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert [int(r['scenario_seed']) for r in rows] == [10000, 10001]
    assert all(float(r['decision_ms']) >= 0 for r in rows)
    assert 'weighted_freshness' in rows[0] and 'total_losses' in rows[0]


def test_training_and_evaluation_seed_overlap_is_rejected():
    with pytest.raises(ValueError, match='overlap'):
        validate_seed_split(0, 3, [2, 10000])
    validate_seed_split(0, 3, [10000])


def test_compare_runs_all_algorithms_and_baselines(tmp_path):
    main(['compare', '--grid', '8', '--horizon', '6', '--episodes', '1',
          '--training-seeds', '2', '--scenario-seeds', '10000',
          '--warmup-steps', '4', '--batch-size', '4', '--planner-candidates', '2',
          '--planner-horizon', '2', '--planner-samples', '1', '--out', str(tmp_path)])
    with (tmp_path/'comparison.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert {r['policy'] for r in rows} == {
        'random', 'sweep', 'greedy', 'planner', 'dqn', 'reinforce', 'a2c'}
    assert all((tmp_path/a/'seed-2'/'policy.pt').exists() for a in ['dqn', 'reinforce', 'a2c'])


def test_train_cli_resume_and_learning_plot(tmp_path):
    out = tmp_path/'train'
    args = ['train', '--algorithm', 'a2c', '--grid', '8', '--horizon', '6',
            '--seed', '3', '--checkpoint-interval', '1', '--progress-interval', '0',
            '--device', 'cpu', '--validation-seeds', '5000', '--validation-interval', '1',
            '--out', str(out)]
    main(args+['--episodes', '1'])
    main(args+['--episodes', '2', '--resume', str(out/'training_state.pt')])
    main(['plot-learning', '--csv', str(out/'training.csv'), '--out', str(tmp_path/'curve.png')])
    rows = list(csv.DictReader((out/'training.csv').open()))
    assert len(rows) == 2 and [int(r['episode']) for r in rows] == [1, 2]
    with Image.open(tmp_path/'curve.png') as image:
        assert image.width > 500 and image.height > 200


def test_compare_rejects_validation_test_overlap_before_writing(tmp_path):
    with pytest.raises(ValueError, match='validation'):
        main(['compare', '--episodes', '1', '--validation-seeds', '5000',
              '--scenario-seeds', '5000', '--out', str(tmp_path/'out')])
    assert not (tmp_path/'out').exists()


def test_compare_preflights_all_resume_settings_before_advancing_any_run(tmp_path):
    from dataclasses import replace

    import torch

    from wild_drone.learning import TrainConfig, train
    config = EnvConfig(grid_size=8, horizon=2)
    settings = TrainConfig(episodes=1, warmup_steps=4, batch_size=4)
    train('dqn', config, settings, tmp_path/'dqn'/'seed-0')
    train('dqn', config, replace(settings, seed=1, gamma=0.5), tmp_path/'dqn'/'seed-1')
    with pytest.raises(ValueError, match='resume'):
        main(['compare', '--grid', '8', '--horizon', '2', '--episodes', '2',
              '--training-seeds', '0', '1', '--warmup-steps', '4', '--batch-size', '4',
              '--resume-auto', '--scenario-seeds', '6000', '--out', str(tmp_path)])
    state = torch.load(tmp_path/'dqn'/'seed-0'/'training_state.pt', weights_only=True)
    assert state['completed_episodes'] == 1
