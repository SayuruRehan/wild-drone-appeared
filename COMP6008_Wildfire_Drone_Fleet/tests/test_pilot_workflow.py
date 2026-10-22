import csv
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from wild_drone.config import EnvConfig


def runner():
    path = Path(__file__).parents[1]/'scripts'/'run_validation_pilot.py'
    spec = importlib.util.spec_from_file_location('pilot', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_validation_runner_resumes_all_methods_and_preserves_protocol_on_mismatch(tmp_path):
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'environment': EnvConfig(grid_size=8, horizon=6).to_dict(),
                                  'training': {'warmup_steps': 4, 'batch_size': 4}}))
    out = tmp_path/'pilot'
    args = ['--config', str(config), '--training-seeds', '2', '--validation-seeds', '5000',
            '--out', str(out)]
    main = runner()
    main(args+['--episodes', '1'])
    main(args+['--episodes', '2'])
    rows = list(csv.DictReader((out/'validation_comparison.csv').open()))
    assert len(rows) == 10
    for algorithm in ['dqn', 'reinforce', 'a2c']:
        state = torch.load(out/algorithm/'seed-2'/'training_state.pt', weights_only=True)
        assert state['completed_episodes'] == 2
        assert (out/f'{algorithm}-learning.png').exists()
    protocol = (out/'protocol.json').read_text()
    raw = json.loads(config.read_text())
    raw['training']['gamma'] = 0.5
    config.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='resume'):
        main(args+['--episodes', '3'])
    assert (out/'protocol.json').read_text() == protocol


def test_dqn_only_variant_records_settings_and_rejects_changed_target(tmp_path):
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'environment': EnvConfig(grid_size=8, horizon=6).to_dict(),
                                  'training': {'warmup_steps': 4, 'batch_size': 4}}))
    out = tmp_path/'double'
    args = ['--config', str(config), '--episodes', '1', '--training-seeds', '0',
            '--validation-seeds', '5000', '--algorithms', 'dqn', '--learning-rate', '0.00003',
            '--epsilon-decay-steps', '40', '--out', str(out)]
    main = runner()
    main(args+['--double-dqn'])
    state = torch.load(out/'dqn'/'seed-0'/'training_state.pt', weights_only=True)
    assert state['train_config']['double_dqn']
    assert state['train_config']['learning_rate'] == 0.00003
    assert state['train_config']['epsilon_decay_steps'] == 40
    assert not (out/'reinforce').exists() and not (out/'a2c').exists()
    before = (out/'protocol.json').read_bytes()
    with pytest.raises(ValueError, match='double_dqn'):
        main(args+['--no-double-dqn'])
    assert (out/'protocol.json').read_bytes() == before
    with pytest.raises(ValueError, match='epsilon_decay_steps'):
        main(args+['--double-dqn', '--epsilon-decay-steps', '80'])
    assert (out/'protocol.json').read_bytes() == before


@pytest.mark.parametrize('changed', [
    ['--training-seeds', '1'], ['--algorithms', 'reinforce']])
def test_existing_study_cannot_be_relabelled_with_other_fits(tmp_path, changed):
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'environment': EnvConfig(grid_size=8, horizon=2).to_dict()}))
    out = tmp_path/'pilot'
    args = ['--config', str(config), '--episodes', '1', '--training-seeds', '0',
            '--validation-seeds', '5000', '--algorithms', 'dqn', '--out', str(out)]
    main = runner()
    main(args)
    before = {name: (out/name).read_bytes() for name in (
        'protocol.json', 'metadata.json', 'validation_comparison.csv')}
    with pytest.raises(ValueError, match='study protocol'):
        main(args+changed)
    assert all((out/name).read_bytes() == content for name, content in before.items())


def test_colab_notebook_code_compiles_without_saved_credentials_or_outputs():
    path = Path(__file__).parents[1]/'notebooks'/'colab_experiments.ipynb'
    notebook = json.loads(path.read_text())
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            assert cell['outputs'] == [] and cell['execution_count'] is None
            compile(''.join(cell['source']), f'{path}:cell{i}', 'exec')


def test_unavailable_device_does_not_replace_existing_study_records(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'environment': EnvConfig(grid_size=8, horizon=2).to_dict()}))
    out = tmp_path/'pilot'
    out.mkdir()
    files = ['protocol.json', 'metadata.json', 'validation_comparison.csv']
    for name in files:
        (out/name).write_text('existing study record')
    with pytest.raises(ValueError, match='CUDA'):
        runner()(['--config', str(config), '--episodes', '1', '--training-seeds', '0',
                  '--device', 'cuda', '--out', str(out)])
    assert all((out/name).read_text() == 'existing study record' for name in files)


def test_colab_benchmark_and_selection_work_without_cuda(tmp_path, monkeypatch):
    import ast
    import sys
    import tempfile
    from importlib.machinery import ModuleSpec
    from types import ModuleType

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    # Rendering a DataFrame is unrelated to device behavior; no pandas test dependency.
    pandas = ModuleType('pandas')
    pandas.__spec__ = ModuleSpec('pandas', loader=None)
    pandas.DataFrame = lambda rows: rows
    monkeypatch.setitem(sys.modules, 'pandas', pandas)
    config_dir = tmp_path/'configs'
    config_dir.mkdir()
    (config_dir/'development.json').write_text(json.dumps({
        'environment': EnvConfig(grid_size=8, horizon=2).to_dict(),
        'training': {'warmup_steps': 4, 'batch_size': 4, 'replay_size': 32}}))
    path = Path(__file__).parents[1]/'notebooks'/'colab_experiments.ipynb'
    notebook = json.loads(path.read_text())
    namespace = {'REPO': tmp_path, 'torch': torch, 'Path': Path, 'tempfile': tempfile,
                 'display': lambda _: None}
    exec(compile(''.join(notebook['cells'][6]['source']), 'colab-benchmark-cell', 'exec'),
         namespace)
    rows = namespace['benchmark_rows']
    assert len(rows) == 3 and {r['device'] for r in rows} == {'cpu'}
    assert all(r['steps'] == 10 and r['steps_per_second'] > 0 for r in rows)
    code = ast.parse(''.join(notebook['cells'][8]['source']))
    selection = next(node.value for node in code.body if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'DEVICE' for t in node.targets))
    device = eval(compile(ast.Expression(selection), 'colab-device-selection', 'eval'), namespace)
    assert device == 'cpu'
