import numpy as np
import pytest
import torch

from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.learning import FleetNetwork, tensor_observations


def probe():
    from wild_drone import centered_inspection
    return centered_inspection


def scene():
    env = WildfireFleetEnv(EnvConfig(grid_size=8, horizon=30, inspection_deadline=10))
    env.reset(seed=17)
    env.positions[:] = [[2, 2], [4, 5], [6, 1]]
    env.ages[:] = np.arange(64).reshape(8, 8) % 15
    return env


def test_current_transform_and_network_preserve_all_inputs_and_parameters_exactly():
    env = scene()
    maps, vector = tensor_observations(env._observation())
    torch.manual_seed(5)
    original = FleetNetwork(8)
    torch.manual_seed(5)
    model = probe().CenteredInspectionNetwork(env.config, 'current')
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.state_dict().items())
    np.testing.assert_array_equal(original(maps, vector)[0].detach(),
                                  model(maps, vector)[0].detach())
    transformed = probe().transform_maps(maps, vector, env.config, 'current')
    assert torch.equal(transformed, maps)


def test_derived_field_matches_observation_cell_urgency_without_action_target_input():
    env = scene()
    maps, vector = tensor_observations(env._observation())
    absolute = probe().transform_maps(maps, vector, env.config, 'absolute')
    field = env.priority_weights()/9*np.minimum((env.ages+1)/10, 1)
    for channel in [5, 6, 7]:
        np.testing.assert_allclose(absolute[0, channel], field, atol=1e-7)
    assert torch.equal(absolute[:, :5], maps[:, :5])
    assert torch.equal(absolute[:, 8], maps[:, 8])


def test_centering_places_each_drone_at_fixed_center_zero_pads_without_wrapping():
    env = scene()
    maps, vector = tensor_observations(env._observation())
    absolute = probe().transform_maps(maps, vector, env.config, 'absolute')
    centered = probe().transform_maps(maps, vector, env.config, 'centered')
    for drone, (r, c) in enumerate(env.positions):
        assert centered[0, 5+drone, 4, 4] == absolute[0, 5+drone, r, c]
        for rr in range(8):
            for cc in range(8):
                sr, sc = rr-4+r, cc-4+c
                value = absolute[0, 5+drone, sr, sc] if 0 <= sr < 8 and 0 <= sc < 8 else 0
                assert centered[0, 5+drone, rr, cc] == value
    assert torch.equal(centered[:, :5], maps[:, :5])
    assert torch.equal(centered[:, 8], maps[:, 8])
    assert torch.all((centered >= 0) & (centered <= 1))


def test_lost_drone_planes_zero_and_transform_is_batched_and_nonmutating():
    env = scene()
    env.lost[1] = True
    maps, vector = tensor_observations(env._observation())
    before, before_vector = maps.clone(), vector.clone()
    for arm in ('absolute', 'centered'):
        result = probe().transform_maps(maps.repeat(2, 1, 1, 1), vector.repeat(2, 1),
                                        env.config, arm)
        assert not result[:, 6].any()
        assert torch.equal(result[0], result[1])
    assert torch.equal(maps, before) and torch.equal(vector, before_vector)
    with pytest.raises(ValueError):
        probe().CenteredInspectionNetwork(env.config, 'unknown')


def test_matched_fit_initialization_sampler_and_export_identity(tmp_path):
    from wild_drone.inspection_targets import collect_states
    config = EnvConfig(grid_size=8, horizon=10)
    data, _, _ = collect_states(config, [21], [2, 4])
    initial, samplers = {}, []
    for arm in probe().ARMS:
        model, manifest, history, tensors = probe().fit_centered(
            config, data, arm=arm, seed=4, updates=2, batch_size=2, output=tmp_path/arm)
        initial[arm] = tensors
        samplers.append(manifest['sample_sequence_sha256'])
        assert len(history) == 2
        loaded = probe().load_centered_model(tmp_path/arm/'final.pt', config, arm)
        assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())
        with pytest.raises(ValueError):
            probe().load_centered_model(tmp_path/arm/'final.pt', config, 'wrong')
        from wild_drone.learning import load_policy
        with pytest.raises(KeyError):
            load_policy(tmp_path/arm/'final.pt')
    assert len(set(samplers)) == 1
    for arm in probe().ARMS:
        assert all(torch.equal(v, initial[arm][k]) for k, v in initial['current'].items())


def test_current_fit_matches_previous_exact_target_fitter(tmp_path):
    from wild_drone.inspection_targets import collect_states, fit_targets
    config = EnvConfig(grid_size=8, horizon=10)
    data, _, _ = collect_states(config, [22], [2, 4])
    previous, _, rows, _ = fit_targets(config, data, seed=5, updates=3,
                                      batch_size=2, output=tmp_path/'previous')
    current, _, current_rows, _ = probe().fit_centered(
        config, data, arm='current', seed=5, updates=3,
        batch_size=2, output=tmp_path/'current')
    assert rows == current_rows
    assert all(torch.equal(v, current.state_dict()[k])
               for k, v in previous.state_dict().items())


def test_runner_exports_all_matched_arms_and_replays_every_prediction(tmp_path, monkeypatch):
    import importlib.util
    import json
    from pathlib import Path

    from wild_drone.experiments import write_rows
    from wild_drone.inspection_targets import collect_states, fit_targets, score_model

    script = Path(__file__).resolve().parents[1]/'scripts'/'run_centered_inspection.py'
    spec = importlib.util.spec_from_file_location('centered_runner', script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config = EnvConfig(grid_size=8, horizon=10)
    training, records, _ = collect_states(config, [21], [2, 4])
    index = [record | {'gains': training['gains'][i].tolist(),
                       'mask': training['masks'][i].tolist()} for i, record in enumerate(records)]
    datasets = {split: training for split in ('training', 'heldout', 'probes')}
    indices = {split: index for split in datasets}
    plan = {'pairs': [{'case': 'identical', 'kind': 'noop', 'before': 0, 'after': 0}]}
    monkeypatch.setattr(runner, 'load_inputs', lambda _: (config, datasets, indices, {}, plan))
    monkeypatch.setattr(runner, 'FRESH_WORLDS', (22,))
    monkeypatch.setattr(runner, 'STEPS', (2,))
    fitter = runner.fit_centered
    # This fixture only tests orchestration; it is not experimental evidence.
    monkeypatch.setattr(runner, 'fit_centered', lambda cfg, data, **kw: fitter(
        cfg, data, **(kw | {'updates': 2, 'batch_size': 2})))
    parent, output = tmp_path/'parent', tmp_path/'output'
    for seed in (0, 1, 2):
        _, _, rows, _ = fit_targets(config, training, seed=seed, updates=2, batch_size=2,
                                    output=parent/f'seed-{seed}')
        write_rows(parent/f'seed-{seed}'/'training.csv', rows)
    runner.main(['--parent-root', str(parent), '--out', str(output),
                 '--source-revision', 'a'*40])
    matching = json.loads((output/'matching.json').read_text())
    assert len(matching) == 3
    assert all(r['current_final_and_training_history_match_parent'] for r in matching)
    complete = json.loads((output/'completion.json').read_text())
    assert len(complete['per_seed_criterion']) == 3
    for arm in probe().ARMS:
        for seed in (0, 1, 2):
            for phase in ('initial', 'final'):
                directory = output/arm/f'seed-{seed}'
                loaded = probe().load_centered_model(directory/f'{phase}.pt', config, arm)
                with np.load(directory/f'{phase}-scores.npz', allow_pickle=False) as scores:
                    for split, data in datasets.items():
                        np.testing.assert_array_equal(scores[split], score_model(loaded, data))
