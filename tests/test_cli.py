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
