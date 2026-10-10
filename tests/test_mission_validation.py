import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.experiments import run_episode
from wild_drone.masked_learning import MaskedRandomPolicy, action_mask


def runner():
    path = Path(__file__).resolve().parents[1]/'scripts'/'run_mission_validation.py'
    assert path.is_file(), 'mission-validation runner has not been implemented'
    spec = importlib.util.spec_from_file_location('mission_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_score_policy_masks_first_argmax_and_rejects_nonfinite_scores():
    module = runner()
    config = EnvConfig(grid_size=8, horizon=12)
    env = WildfireFleetEnv(config)
    obs, _ = env.reset(seed=12)
    values = np.zeros(125, dtype=np.float32)
    forbidden = np.flatnonzero(~action_mask(env))[0]
    values[forbidden] = 100
    policy = module.ScorePolicy(config, scores=values)
    assert policy.act(env, obs) == np.flatnonzero(action_mask(env))[0]
    assert np.array_equal(policy.last_scores, values)
    assert np.array_equal(policy.last_mask, action_mask(env))
    values[0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        module.ScorePolicy(config, scores=values).act(env, obs)
    other = WildfireFleetEnv(EnvConfig(grid_size=8, horizon=13))
    with pytest.raises(ValueError, match='configuration'):
        policy.act(other, other.reset(seed=12)[0])


def test_trace_episode_matches_existing_full_horizon_metrics():
    module = runner()
    config = EnvConfig(grid_size=8, horizon=12, battery_capacity=5)
    actual, trace, scores = module.rollout(config, MaskedRandomPolicy(17), 17,
                                          name='random')
    expected = run_episode(config, MaskedRandomPolicy(17), 17, name='random')
    for key, value in expected.items():
        if key != 'decision_ms':
            assert actual[key] == pytest.approx(value) if isinstance(value, float) else \
                actual[key] == value
    assert len(trace) == actual['steps']
    assert scores is None
    assert trace[0]['pre_sha256'] != trace[0]['post_sha256']
    assert trace[-1]['total_losses'] == actual['total_losses']
    assert sum(r['energy_used'] for r in trace) == actual['energy_used']
    assert actual['weighted_freshness'] == sum(r['weighted_freshness'] for r in trace)/12
    assert actual['overdue_fraction'] == pytest.approx(
        (sum(r['overdue_fraction'] for r in trace)+(12-len(trace)))/12)


def test_failed_fleet_receives_full_horizon_padding():
    module = runner()
    config = EnvConfig(grid_size=8, horizon=12, battery_capacity=1)

    class UnsafePolicy:
        def act(self, env, observation):
            return 38  # Three distinct moves away from base exhaust all batteries.

    episode, trace, _ = module.rollout(config, UnsafePolicy(), 7, name='unsafe_fixture')
    assert episode['steps'] == 1 and episode['fleet_failure'] == 1
    assert episode['total_losses'] == 3 and episode['intact_fleet'] == 0
    assert episode['weighted_freshness'] == trace[0]['weighted_freshness']/12
    assert episode['overdue_fraction'] == (trace[0]['overdue_fraction']+11)/12
    expected = run_episode(config, UnsafePolicy(), 7, name='unsafe_fixture')
    assert episode['return'] == expected['return']


def test_mission_criteria_require_each_seed_and_keep_baseline_check_separate():
    module = runner()
    rows = [dict(policy='greedy', training_seed=None, scenario_seed=w,
                 weighted_freshness=.3, overdue_fraction=.5, total_losses=0,
                 fleet_failure=0) for w in (20, 21)]
    for seed in (0, 1, 2):
        for corpus, fresh in [('small', .2), ('expanded', .28)]:
            rows += [dict(policy=corpus, training_seed=seed, scenario_seed=w,
                          weighted_freshness=fresh, overdue_fraction=.52,
                          total_losses=0, fleet_failure=0) for w in (20, 21)]
    result = module.decisions(rows)
    assert all(r['data_effect_passed'] and r['greedy_capability_passed'] for r in result)
    rows[-1]['weighted_freshness'] = .01
    rows[-1]['fleet_failure'] = 1
    result = module.decisions(rows)
    assert result[0]['data_effect_passed'] and not result[2]['data_effect_passed']
    assert not result[2]['greedy_capability_passed']


def test_checkpoint_preflight_rejects_tampered_and_wrong_seed_exports(tmp_path):
    import hashlib
    import json

    from wild_drone.centered_inspection import FORMAT, CenteredInspectionNetwork

    module = runner()
    config = EnvConfig(grid_size=8, horizon=12)
    launch = {'env_config': config.to_dict(), 'source_sha256': {}}
    (tmp_path/'launch.json').write_text(json.dumps(launch))
    index = {}
    for corpus in ('small', 'expanded'):
        for seed in (0, 1, 2):
            p = tmp_path/corpus/f'seed-{seed}'/'final.pt'
            p.parent.mkdir(parents=True)
            torch.save(dict(format=FORMAT, env_config=config.to_dict(), arm='centered',
                            seed=seed, completed_updates=2000,
                            state_dict=CenteredInspectionNetwork(config, 'centered').state_dict()),
                       p)
            index[f'{corpus}/{seed}'] = {'checkpoints': {'final.pt': {
                'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}}}
    (tmp_path/'checkpoints.json').write_text(json.dumps(index))
    loaded, identities = module.load_models(tmp_path, archive=tmp_path)
    assert loaded == config and len(identities) == 6
    p.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='identity'):
        module.load_models(tmp_path, archive=tmp_path)
    torch.save(dict(format=FORMAT, env_config=config.to_dict(), arm='centered',
                    seed=1, completed_updates=2000,
                    state_dict=CenteredInspectionNetwork(config, 'centered').state_dict()), p)
    index['expanded/2']['checkpoints']['final.pt'].update(
        sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)
    (tmp_path/'checkpoints.json').write_text(json.dumps(index))
    with pytest.raises(ValueError, match='metadata'):
        module.load_models(tmp_path, archive=tmp_path)


def test_end_to_end_keeps_models_fixed_and_exports_every_score(tmp_path, monkeypatch):
    import csv
    import gzip
    import hashlib
    import json

    from wild_drone.centered_inspection import FORMAT, CenteredInspectionNetwork

    module = runner()
    config = EnvConfig(grid_size=8, horizon=3)
    parent = tmp_path/'parent'
    parent.mkdir()
    (parent/'launch.json').write_text(json.dumps({'env_config': config.to_dict(),
                                                 'source_sha256': {}}))
    index = {}
    for corpus in ('small', 'expanded'):
        for seed in (0, 1, 2):
            path = parent/corpus/f'seed-{seed}'/'final.pt'
            path.parent.mkdir(parents=True)
            torch.manual_seed(seed)
            torch.save(dict(format=FORMAT, env_config=config.to_dict(), arm='centered',
                            seed=seed, completed_updates=2000,
                            state_dict=CenteredInspectionNetwork(config, 'centered').state_dict()),
                       path)
            index[f'{corpus}/{seed}'] = {'checkpoints': {'final.pt': {
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'bytes': path.stat().st_size}}}
    (parent/'checkpoints.json').write_text(json.dumps(index))
    controls = {c: {'scores': np.arange(125).tolist()} for c in ('small', 'expanded')}
    (parent/'action-mean-controls.json').write_text(json.dumps(controls))
    (parent/'checksums.json').write_text(json.dumps({'action-mean-controls.json': {
        'sha256': hashlib.sha256((parent/'action-mean-controls.json').read_bytes()).hexdigest()}}))
    monkeypatch.setattr(module, 'PARENT', parent)
    monkeypatch.setattr(module, 'WORLDS', (6600, 6601))
    output = tmp_path/'output'
    argv = ['--parent-root', str(parent), '--out', str(output), '--source-revision', 'a'*40]
    module.main(argv)
    episodes = list(csv.DictReader((output/'episodes.csv').open()))
    with gzip.open(output/'steps.csv.gz', 'rt') as f:
        trace = list(csv.DictReader(f))
    assert len(episodes) == 22 and len(trace) == 66
    assert len(list(output.glob('*scores.npz'))) == 8
    for path in output.glob('*scores.npz'):
        with np.load(path, allow_pickle=False) as data:
            assert data['scores'].shape == (6, 125)
            chosen = np.where(data['masks'], data['scores'], -np.inf).argmax(1)
            name = path.name.split('-seed-')[0]
            seed = path.name.split('-seed-')[1].split('-scores')[0]
            group = [r for r in trace if r['policy'] == name and
                     r['training_seed'] == ('' if seed == 'None' else seed)]
            assert chosen.tolist() == [int(r['action']) for r in group]
    for key, entry in index.items():
        corpus, seed = key.split('/')
        path = parent/corpus/f'seed-{seed}'/'final.pt'
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        assert checksum == entry['checkpoints']['final.pt']['sha256']
    assert json.loads((output/'completion.json').read_text())['rl_updates'] == 0
    with pytest.raises(FileExistsError):
        module.main(argv)
