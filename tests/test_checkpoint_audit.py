from dataclasses import replace

import pytest
import torch

from wild_drone.checkpoint_audit import audit_checkpoints
from wild_drone.config import EnvConfig
from wild_drone.learning import TrainConfig, train


@pytest.fixture
def trained(tmp_path):
    config = EnvConfig(grid_size=8, horizon=6)
    settings = TrainConfig(episodes=2, validation_seeds=(5000,), validation_interval=1,
                           checkpoint_interval=1, progress_interval=0)
    train('dqn', config, settings, tmp_path)
    return tmp_path, config, settings


def test_audit_records_verified_final_and_selected_boundaries(trained):
    out, _, _ = trained
    report = audit_checkpoints(out)
    assert report['completed_episodes'] == 2
    assert report['exports']['policy.pt']['completed_episodes'] == 2
    state = torch.load(out/'training_state.pt', weights_only=True)
    assert report['exports']['best_policy.pt']['completed_episodes'] == (
        state['best_checkpoint']['completed_episodes'])
    assert len(report['authority_sha256']) == 64


@pytest.mark.parametrize('damage', ['episode', 'weights', 'selection', 'missing'])
def test_audit_rejects_stale_or_damaged_export(trained, damage):
    out, _, _ = trained
    name = 'best_policy.pt' if damage == 'selection' else 'policy.pt'
    path = out/name
    data = torch.load(path, weights_only=True)
    if damage == 'episode':
        data['completed_episodes'] = 1
    elif damage == 'weights':
        # Same episode and metadata, but another network's tensors.
        data['state_dict']['head.bias'][0] += 1
    elif damage == 'selection':
        data['selection']['validation_return'] += 1
    else:
        path.unlink()
    if damage != 'missing':
        torch.save(data, path)
    with pytest.raises(ValueError, match=name):
        audit_checkpoints(out)


def test_noop_resume_recovers_exports_from_authority(trained):
    out, config, settings = trained
    (out/'policy.pt').unlink()
    train('dqn', config, settings, out, resume_from=out/'training_state.pt')
    assert audit_checkpoints(out)['completed_episodes'] == 2
    assert (out/'checkpoint_manifest.json').exists()


def test_unreadable_export_is_reported_with_filename(trained):
    out, _, _ = trained
    (out/'policy.pt').write_bytes(b'not a torch checkpoint')
    with pytest.raises(ValueError, match='policy.pt: cannot read checkpoint'):
        audit_checkpoints(out)


def test_notebook_archive_stops_on_stale_drive_export(trained, monkeypatch):
    import json
    import sys
    from pathlib import Path
    from types import ModuleType, SimpleNamespace

    out, _, _ = trained
    policy = torch.load(out/'policy.pt', weights_only=True)
    policy['completed_episodes'] = 1
    torch.save(policy, out/'policy.pt')
    downloads = []
    colab = ModuleType('google.colab')
    colab.files = SimpleNamespace(download=downloads.append)
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    path = Path(__file__).parents[1]/'notebooks'/'colab_experiments.ipynb'
    notebook = json.loads(path.read_text())
    code = ''.join(notebook['cells'][-1]['source'])
    with pytest.raises(ValueError, match='policy.pt'):
        exec(compile(code, 'colab-archive', 'exec'), {'RUN_ROOT': out, 'Path': Path})
    assert downloads == []


def test_old_standard_dqn_state_resumes_but_cannot_become_double_dqn(trained):
    out, config, settings = trained
    state = torch.load(out/'training_state.pt', weights_only=True)
    del state['train_config']['double_dqn']
    torch.save(state, out/'training_state.pt')
    train('dqn', config, settings, out, resume_from=out/'training_state.pt')
    with pytest.raises(ValueError, match='double_dqn'):
        train('dqn', config, replace(settings, double_dqn=True), out,
              resume_from=out/'training_state.pt')
