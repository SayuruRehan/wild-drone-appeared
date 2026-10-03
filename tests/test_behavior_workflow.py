import csv
import importlib.util
from pathlib import Path

import pytest

from wild_drone.config import EnvConfig
from wild_drone.learning import TrainConfig, train


def runner():
    path = Path(__file__).parents[1]/'scripts'/'diagnose_behavior.py'
    spec = importlib.util.spec_from_file_location('behavior_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_diagnostic_workflow_preflights_and_writes_audited_missions(tmp_path):
    directory, out = tmp_path/'fit', tmp_path/'diagnostics'
    train('dqn', EnvConfig(grid_size=8, horizon=6), TrainConfig(
        episodes=2, validation_seeds=(5000,), progress_interval=0), directory)
    main = runner()
    main([str(directory), '--scenario-seeds', '5000', '--out', str(out)])
    rows = list(csv.DictReader((out/'missions.csv').open()))
    assert len(rows) == 6
    learned = [r for r in rows if r['policy'] == 'dqn']
    assert {r['checkpoint'] for r in learned} == {'policy.pt', 'best_policy.pt'}
    assert all(len(r['checkpoint_sha256']) == 64 for r in learned)
    assert {r['policy'] for r in rows} == {'dqn', 'random', 'sweep', 'greedy', 'park'}
    trace = list(csv.DictReader((out/'example-trace.csv').open()))
    assert len(trace) == 3*int(learned[0]['steps'])
    assert {r['drone'] for r in trace[-3:]} == {'0', '1', '2'}
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(ValueError, match='overlap'):
        main([str(directory), '--scenario-seeds', '0', '--out', str(out)])
    assert all((out/name).read_bytes() == data for name, data in before.items())


def test_cli_records_exploration_override(tmp_path):
    import torch

    from wild_drone.cli import main
    main(['train', '--algorithm', 'dqn', '--episodes', '1', '--grid', '8', '--horizon', '2',
          '--epsilon-decay-steps', '40', '--out', str(tmp_path)])
    state = torch.load(tmp_path/'training_state.pt', weights_only=True)
    assert state['train_config']['epsilon_decay_steps'] == 40
