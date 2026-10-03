from dataclasses import replace

import pytest
import torch

from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.imitation_probe import (
    ProbeConfig,
    collect_teacher_data,
    fit_probe,
    load_probe,
)


def test_teacher_collection_reproduces_real_states_and_keeps_scenario_ids():
    config = EnvConfig(grid_size=8, horizon=6)
    a, b = (collect_teacher_data(config, (0, 1)) for _ in range(2))
    assert len(a['actions']) == 12
    assert set(a['scenario_seeds'].tolist()) == {0, 1}
    assert a['critical'].shape == (12, 3)
    assert a['maps'].shape == (12, 9, 8, 8)
    for key in a:
        if isinstance(a[key], torch.Tensor):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        else:
            assert a[key] == b[key]


def test_probe_roundtrip_preserves_predictions_and_is_explicitly_not_dqn(tmp_path):
    config = EnvConfig(grid_size=8, horizon=6)
    training = collect_teacher_data(config, (0, 1))
    validation = collect_teacher_data(config, (5000,))
    settings = ProbeConfig(epochs=2, batch_size=4, evaluation_interval=1)
    report = fit_probe(config, training, validation, settings, tmp_path)
    assert report['method'] == 'supervised-greedy-imitation-probe'
    assert report['completed_epochs'] == 2
    policy = load_probe(tmp_path/'final_probe.pt')
    duplicate = load_probe(tmp_path/'final_probe.pt')
    env = WildfireFleetEnv(config)
    obs, _ = env.reset(seed=5000)
    assert policy.act(env, obs) == duplicate.act(env, obs)
    assert (tmp_path/'probe_manifest.json').exists()
    from wild_drone.learning import load_policy
    with pytest.raises(KeyError):
        load_policy(tmp_path/'final_probe.pt')


def test_validation_labels_do_not_change_final_training_weights(tmp_path):
    config = EnvConfig(grid_size=8, horizon=6)
    training = collect_teacher_data(config, (0, 1))
    validation = collect_teacher_data(config, (5000,))
    changed = {key: value.clone() if isinstance(value, torch.Tensor) else value
               for key, value in validation.items()}
    changed['actions'] = (changed['actions']+1) % 125
    settings = ProbeConfig(epochs=2, batch_size=4, evaluation_interval=1)
    fit_probe(config, training, validation, settings, tmp_path/'a')
    fit_probe(config, training, changed, settings, tmp_path/'b')
    a, b = (load_probe(tmp_path/p/'final_probe.pt').model.state_dict() for p in ('a', 'b'))
    assert all(torch.equal(a[key], b[key]) for key in a)


def test_probe_overlap_and_existing_outputs_are_rejected_before_writing(tmp_path):
    config = EnvConfig(grid_size=8, horizon=6)
    training = collect_teacher_data(config, (0,))
    with pytest.raises(ValueError, match='overlap'):
        fit_probe(config, training, training, ProbeConfig(epochs=1), tmp_path/'overlap')
    assert not (tmp_path/'overlap').exists()
    out = tmp_path/'existing'
    out.mkdir()
    (out/'sentinel.txt').write_text('preserve')
    with pytest.raises(ValueError, match='existing'):
        fit_probe(config, training, collect_teacher_data(config, (5000,)),
                  ProbeConfig(epochs=1), out)
    assert (out/'sentinel.txt').read_text() == 'preserve'


def test_probe_rejects_invalid_settings_and_other_checkpoint_families(tmp_path):
    with pytest.raises(ValueError):
        ProbeConfig(epochs=0)
    with pytest.raises(ValueError):
        replace(ProbeConfig(), learning_rate=float('nan'))
    torch.save({'format': 'dqn'}, tmp_path/'wrong.pt')
    with pytest.raises(ValueError, match='imitation probe'):
        load_probe(tmp_path/'wrong.pt')


def test_export_rejects_silent_tensor_corruption(tmp_path, monkeypatch):
    import wild_drone.imitation_probe as probe
    original = probe.atomic_torch_save

    def corrupt(data, path):
        original(data, path)
        saved = torch.load(path, weights_only=True)
        next(iter(saved['state_dict'].values())).flatten()[0] += 1
        original(saved, path)

    monkeypatch.setattr(probe, 'atomic_torch_save', corrupt)
    training = collect_teacher_data(EnvConfig(grid_size=8, horizon=6), (0,))
    validation = collect_teacher_data(EnvConfig(grid_size=8, horizon=6), (5000,))
    with pytest.raises(ValueError, match='tensors differ'):
        fit_probe(EnvConfig(grid_size=8, horizon=6), training, validation,
                  ProbeConfig(epochs=1), tmp_path)
    assert not (tmp_path/'probe_manifest.json').exists()


def test_runner_preserves_identity_and_preflights_before_collecting(tmp_path, monkeypatch):
    import hashlib
    import importlib.util
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1]/'scripts/run_imitation_probe.py'
    spec = importlib.util.spec_from_file_location('probe_runner', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    out = tmp_path/'probe'
    config = EnvConfig(grid_size=8, horizon=6)
    runner.run_probe(config, [0, 1], [5000], [0], ProbeConfig(epochs=1), out)
    protocol = json.loads((out/'protocol.json').read_text())
    for record in protocol['datasets'].values():
        assert hashlib.sha256((out/record['file']).read_bytes()).hexdigest() == record['sha256']
    import csv
    rows = list(csv.DictReader((out/'validation_missions.csv').open()))
    assert len(rows) == 4
    manifest = json.loads((out/'seed-0/probe_manifest.json').read_text())
    for row in rows[:3]:
        assert row['checkpoint_sha256'] == manifest[row['checkpoint']]['sha256']
    assert rows[-1]['policy'] == 'greedy'

    def forbidden(*args):
        pytest.fail('preflight must reject before expensive collection')

    monkeypatch.setattr(runner, 'collect_teacher_data', forbidden)
    with pytest.raises(ValueError, match='existing'):
        runner.run_probe(config, [0], [5000], [0], ProbeConfig(epochs=1), out)
    with pytest.raises(ValueError, match='overlap'):
        runner.run_probe(config, [0], [0], [0], ProbeConfig(epochs=1), tmp_path/'overlap')
    assert not (tmp_path/'overlap').exists()


def test_label_metrics_distinguish_critical_returns_from_charging_and_lost_commands():
    from wild_drone.environment import encode_action
    from wild_drone.imitation_probe import label_metrics

    data = collect_teacher_data(EnvConfig(grid_size=8, horizon=1), (0,))
    data['positions'][0] = torch.tensor([[4, 5], [4, 4], [4, 6]])
    data['base'][0] = torch.tensor([4, 4])
    data['live'][0] = torch.tensor([True, True, False])
    data['critical'][0] = torch.tensor([True, False, False])
    data['needs_charge'][0] = torch.tensor([False, True, False])
    # MOVES[3] is west; the lost drone's disagreement should not affect live accuracy.
    data['actions'][0] = encode_action([3, 0, 1])

    class Fixed(torch.nn.Module):
        def forward(self, maps, vector):
            logits = torch.zeros(len(maps), 125)
            logits[:, encode_action([3, 0, 2])] = 10
            return logits, None

    metrics = label_metrics(Fixed(), data, 1)
    assert metrics['joint_accuracy'] == 0
    assert metrics['live_command_accuracy'] == 1
    assert metrics['critical_command_accuracy'] == 1
    assert metrics['critical_toward_base_command_fraction'] == 1
    assert metrics['charging_hold_fraction'] == 1
    data['critical'].fill_(False)
    assert label_metrics(Fixed(), data, 1)['critical_command_accuracy'] is None
