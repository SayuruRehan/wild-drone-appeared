"""Verify local binary identities, matching, transforms and newly collected worlds."""

import argparse
import copy
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from wild_drone.baselines import make_policy
from wild_drone.centered_inspection import FORMAT, load_centered_model, transform_maps
from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.inspection_targets import score_model
from wild_drone.learning import FleetNetwork, load_policy, tensor_observations
from wild_drone.masked_learning import MaskedRandomPolicy, action_mask, load_masked_policy


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--centered-root', type=Path, required=True)
    args = parser.parse_args()
    root, run = Path(__file__).resolve().parents[3], args.run
    archive = Path(__file__).resolve().parent
    launch = json.loads((run/'launch.json').read_text())
    config = EnvConfig(**launch['env_config'])
    index = json.loads(gzip.decompress((run/'inputs.json.gz').read_bytes()))
    identities = json.loads((run/'datasets.json').read_text())
    checkpoints = json.loads((run/'checkpoints.json').read_text())
    sampling = json.loads((run/'sampling.json').read_text())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for path, identity in launch['source_sha256'].items():
        assert sha(root/path) == identity
    datasets = {}
    for key, identity in identities.items():
        path = run/f'{key}.pt'
        assert path.stat().st_size == identity['bytes'] and sha(path) == identity['sha256']
        datasets[key] = torch.load(path, weights_only=True)
        data = datasets[key]
        for i, record in enumerate(index[key]):
            np.testing.assert_array_equal(record['gains'], data['gains'][i])
            np.testing.assert_array_equal(record['mask'], data['masks'][i])
            ids = np.flatnonzero(record['mask'])
            low, high = min(record['gains'][j] for j in ids), max(record['gains'][j] for j in ids)
            targets = np.zeros(125, dtype=np.float32)
            if high-low > 1e-12:
                targets[ids] = [(record['gains'][j]-low)/(high-low) for j in ids]
            np.testing.assert_array_equal(targets, data['targets'][i])
            assert bool(data['informative'][i]) == (high-low > 1e-12)
        maps, vectors = data['maps'], data['vectors']
        arrays = maps.numpy()
        burning, fuel, priority, age = [arrays[:, j] for j in (0, 1, 3, 4)]
        neighbours = np.zeros_like(burning)
        neighbours[:, 1:] += burning[:, :-1]
        neighbours[:, :-1] += burning[:, 1:]
        neighbours[:, :, 1:] += burning[:, :, :-1]
        neighbours[:, :, :-1] += burning[:, :, 1:]
        frontier = (neighbours > 0) & (fuel > 0) & (burning < .5)
        age_limit = max(config.horizon, config.inspection_deadline+1)
        field = priority*(1+2*burning+frontier)/3*np.minimum(
            (age*age_limit+1)/config.inspection_deadline, 1)
        absolute = transform_maps(maps, vectors, config, 'absolute').numpy()
        centered = transform_maps(maps, vectors, config, 'centered').numpy()
        poses = np.rint(vectors[:, :12].numpy().reshape(-1, 3, 4)[:, :, :2]*19).astype(int)
        alive = vectors[:, :12].numpy().reshape(-1, 3, 4)[:, :, 3] < .5
        for drone in range(3):
            np.testing.assert_allclose(absolute[:, 5+drone], field*alive[:, drone, None, None],
                                       rtol=0, atol=1e-7)
            expected = np.zeros_like(field)
            for i in range(len(maps)):
                if not alive[i, drone]:
                    continue
                r, c = poses[i, drone]
                for row in range(20):
                    for col in range(20):
                        sr, sc = row-10+r, col-10+c
                        if 0 <= sr < 20 and 0 <= sc < 20:
                            expected[i, row, col] = absolute[i, 5+drone, sr, sc]
            np.testing.assert_array_equal(expected, centered[:, 5+drone])
        np.testing.assert_array_equal(centered[:, :5], arrays[:, :5])
        np.testing.assert_array_equal(centered[:, 8], arrays[:, 8])
    old_archive = root/'docs/experiments/inspection-targets-2026-10-10'
    parent_identity = json.loads((old_archive/'datasets.json').read_text())['training']
    original_path = args.parent_root/'training.pt'
    assert sha(original_path) == parent_identity['sha256']
    original = torch.load(original_path, weights_only=True)
    for key, value in original.items():
        assert torch.equal(value, datasets['small'][key])
        assert torch.equal(value, datasets['expanded'][key][:len(value)])
    assert index['expanded'][:len(index['small'])] == index['small']
    worlds = {key: {r['scenario_seed'] for r in records} for key, records in index.items()}
    assert worlds['small'] < worlds['expanded']
    assert not worlds['expanded'] & (worlds['fresh'] | worlds['reused'])
    assert not worlds['fresh'] & worlds['reused']
    score_count = 0
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        initial = FleetNetwork(20).state_dict()
        for corpus in ('small', 'expanded'):
            directory = run/corpus/f'seed-{seed}'
            ids = torch.where(datasets[corpus]['informative'])[0]
            generator = torch.Generator().manual_seed(seed+1000)
            counts = np.zeros(len(datasets[corpus]['maps']), dtype=np.int64)
            sequence = hashlib.sha256()
            for _ in range(2000):
                batch = ids[torch.randint(len(ids), (64,), generator=generator)].numpy()
                sequence.update(batch.astype('<i8').tobytes())
                for i in batch:
                    counts[i] += 1
            metadata, exposure = checkpoints[f'{corpus}/{seed}'], sampling[f'{corpus}/{seed}']
            assert metadata['sample_sequence_sha256'] == sequence.hexdigest()
            assert exposure['sample_sequence_sha256'] == sequence.hexdigest()
            np.testing.assert_array_equal(counts, exposure['per_state_counts'])
            assert exposure['sampled_distinct_states'] == int((counts > 0).sum())
            assert exposure['sample_draws'] == 128000
            assert exposure['informative_states'] == len(ids)
            assert exposure['draws_per_informative_state'] == 128000/len(ids)
            assert metadata['training_dataset_sha256'] == identities[corpus]['sha256']
            assert metadata['parameters'] == sum(v.numel() for v in initial.values()) == 127677
            for phase, updates in (('initial', 0), ('final', 2000)):
                path = directory/f'{phase}.pt'
                identity = metadata['checkpoints'][path.name]
                assert sha(path) == identity['sha256'] and path.stat().st_size == identity['bytes']
                saved = torch.load(path, weights_only=True)
                assert saved['format'] == FORMAT and saved['arm'] == 'centered'
                assert saved['seed'] == seed and saved['completed_updates'] == updates
                assert saved['env_config'] == config.to_dict()
                if phase == 'initial':
                    assert all(torch.equal(v, saved['state_dict'][k]) for k, v in initial.items())
                for loader, error in ((load_policy, KeyError), (load_masked_policy, ValueError)):
                    try:
                        loader(path)
                    except error:
                        pass
                    else:
                        raise AssertionError('supervised weights accepted as RL')
            model = load_centered_model(directory/'final.pt', config, 'centered')
            with np.load(directory/'final-scores.npz', allow_pickle=False) as raw:
                for split in ('training', 'fresh', 'reused'):
                    data = datasets[corpus if split == 'training' else split]
                    np.testing.assert_array_equal(score_model(model, data), raw[split])
                    score_count += len(raw[split])
            if corpus == 'small':
                old = args.centered_root/'centered'/f'seed-{seed}'
                previous = torch.load(old/'final.pt', weights_only=True)
                assert all(torch.equal(v, previous['state_dict'][k])
                           for k, v in model.state_dict().items())
                a = list(csv.DictReader((old/'training.csv').open()))
                b = list(csv.DictReader((directory/'training.csv').open()))
                assert a == b and len(b) == 2000
        model = load_centered_model(run/'small'/f'seed-{seed}'/'initial.pt', config, 'centered')
        with np.load(run/'initial'/f'seed-{seed}-scores.npz', allow_pickle=False) as raw:
            for split in ('fresh', 'reused'):
                np.testing.assert_array_equal(score_model(model, datasets[split]), raw[split])
                score_count += len(raw[split])
    states_checked, rewards_checked = 0, 0
    replay_missing = {}
    added_worlds = launch['expanded_worlds'][len(launch['small_worlds']):]
    for split, selected_worlds in [('expanded', added_worlds),
                                   ('fresh', launch['fresh_worlds'])]:
        lookup = {r['case']: (i, r) for i, r in enumerate(index[split])}
        missing = []
        for world in selected_worlds:
            for collector in ('masked_random', 'greedy'):
                env = WildfireFleetEnv(config)
                observation, _ = env.reset(seed=world)
                policy = MaskedRandomPolicy(world) if collector == 'masked_random' else \
                    make_policy('greedy')
                for step in range(97):
                    if step in launch['snapshot_steps']:
                        i, record = lookup[f'{world}-{collector}-{step}']
                        obs = env._observation()
                        identity = hashlib.sha256(json.dumps(
                            {k: v.tolist() for k, v in obs.items()},
                            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                        assert identity == record['observation_sha256']
                        maps, vectors = tensor_observations(obs)
                        assert torch.equal(maps[0], datasets[split]['maps'][i])
                        assert torch.equal(vectors[0], datasets[split]['vectors'][i])
                        np.testing.assert_array_equal(action_mask(env), record['mask'])
                        gains = [copy.deepcopy(env).step(action)[4][
                            'reward_components']['freshness_gain'] for action in range(125)]
                        np.testing.assert_allclose(gains, record['gains'], rtol=0, atol=1e-13)
                        states_checked += 1
                        rewards_checked += 125
                    if step == 96:
                        break
                    observation, _, terminal, truncated, _ = env.step(policy.act(env, observation))
                    if terminal or truncated:
                        missing.extend({'scenario_seed': world, 'collector': collector, 'step': s,
                            'reason': 'collector terminated before snapshot'}
                            for s in launch['snapshot_steps'] if s > step)
                        break
        replay_missing['appended_training' if split == 'expanded' else 'fresh'] = missing
        print(f'{split}: new worlds replayed', flush=True)
    missing = json.loads((run/'missing-snapshots.json').read_text())
    for key, value in replay_missing.items():
        assert value == missing[key]
    assert states_checked == len(index['expanded'])-len(index['small'])+len(index['fresh'])
    result = {'status': 'PASS', 'source_hashes_verified': len(launch['source_sha256']),
              'dataset_files_verified': 4, 'checkpoint_files_verified': 12,
              'transform_states_checked': sum(len(d['maps']) for d in datasets.values()),
              'independent_field_formula_tolerance': 1e-7, 'centering_padding_exact': True,
              'nested_original_tensors_and_records_exact': True,
              'new_natural_states_replayed': states_checked,
              'original_step_inspection_rewards_checked': rewards_checked,
              'raw_score_vectors_exactly_reproduced': score_count,
              'initial_tensors_and_parameter_counts_matched': True,
              'small_final_tensors_and_6000_losses_match_parent': True,
              'sample_sequences_counts_and_exposure_verified': True,
              'target_normalisation_and_world_splits_verified': True,
              'supervised_checkpoints_rejected_by_rl_loaders': True}
    (archive/'replay-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
