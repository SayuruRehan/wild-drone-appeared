"""Audit local datasets/weights against archived identities and replay fresh worlds.

Run from the repository root with --run and --parent-root pointing to binary
outputs. Numerical CSV recomputation is separately runnable without these files.
"""

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
from wild_drone.centered_inspection import ARMS, FORMAT, load_centered_model, transform_maps
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
    args = parser.parse_args()
    root, run, parent = Path(__file__).resolve().parents[3], args.run, args.parent_root
    archive = Path(__file__).resolve().parent
    launch = json.loads((run/'launch.json').read_text())
    config = EnvConfig(**launch['env_config'])
    index = json.loads(gzip.decompress((run/'inputs.json.gz').read_bytes()))
    identities = json.loads((run/'datasets.json').read_text())
    checkpoints = json.loads((run/'checkpoints.json').read_text())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for path, identity in launch['source_sha256'].items():
        assert sha(root/path) == identity
    datasets = {}
    for split, identity in identities.items():
        path = (run if split == 'fresh' else parent)/f'{split}.pt'
        assert path.stat().st_size == identity['bytes'] and sha(path) == identity['sha256']
        datasets[split] = torch.load(path, weights_only=True)
        data = datasets[split]
        for i, record in enumerate(index[split]):
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
        assert np.all((field >= 0) & (field <= 1))
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
        for result in (absolute, centered):
            np.testing.assert_array_equal(result[:, :5], arrays[:, :5])
            np.testing.assert_array_equal(result[:, 8], arrays[:, 8])
    vectors_checked = 0
    informative_ids = torch.where(datasets['training']['informative'])[0]
    matching = json.loads((run/'matching.json').read_text())
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        original = FleetNetwork(20).state_dict()
        generator = torch.Generator().manual_seed(seed+1000)
        sequence = hashlib.sha256()
        for _ in range(2000):
            batch = informative_ids[torch.randint(len(informative_ids), (64,), generator=generator)]
            sequence.update(batch.numpy().astype('<i8').tobytes())
        assert sequence.hexdigest() == matching[seed]['sample_sequence_sha256']
        for arm in ARMS:
            directory = run/arm/f'seed-{seed}'
            metadata = checkpoints[f'{arm}/{seed}']
            assert metadata['parameters'] == sum(v.numel() for v in original.values())
            assert metadata['sample_sequence_sha256'] == sequence.hexdigest()
            for phase, updates in (('initial', 0), ('final', 2000)):
                path = directory/f'{phase}.pt'
                identity = metadata['checkpoints'][path.name]
                assert sha(path) == identity['sha256'] and path.stat().st_size == identity['bytes']
                saved = torch.load(path, weights_only=True)
                assert saved['format'] == FORMAT and saved['arm'] == arm
                assert saved['seed'] == seed and saved['completed_updates'] == updates
                assert saved['env_config'] == config.to_dict()
                if phase == 'initial':
                    assert all(torch.equal(v, saved['state_dict'][k]) for k, v in original.items())
                model = load_centered_model(path, config, arm)
                with np.load(directory/f'{phase}-scores.npz', allow_pickle=False) as raw:
                    for split, data in datasets.items():
                        np.testing.assert_array_equal(score_model(model, data), raw[split])
                        vectors_checked += len(raw[split])
                for loader, error in ((load_policy, KeyError), (load_masked_policy, ValueError)):
                    try:
                        loader(path)
                    except error:
                        pass
                    else:
                        raise AssertionError('supervised checkpoint accepted as RL')
            if arm == 'current':
                previous = torch.load(parent/f'seed-{seed}'/'final.pt', weights_only=True)
                assert all(torch.equal(v, previous['state_dict'][k])
                           for k, v in model.state_dict().items())
                old = list(csv.DictReader((parent/f'seed-{seed}'/'training.csv').open()))
                new = list(csv.DictReader((directory/'training.csv').open()))
                assert old == new and len(new) == 2000
    lookup = {r['case']: (i, r) for i, r in enumerate(index['fresh'])}
    fresh_count, reward_count = 0, 0
    missing = []
    for world in launch['fresh_worlds']:
        for collector in ('masked_random', 'greedy'):
            env = WildfireFleetEnv(config)
            observation, _ = env.reset(seed=world)
            policy = MaskedRandomPolicy(world) if collector == 'masked_random' else \
                make_policy('greedy')
            for step in range(97):
                if step in launch['snapshot_steps']:
                    i, record = lookup[f'{world}-{collector}-{step}']
                    obs = env._observation()
                    observation_sha = hashlib.sha256(json.dumps(
                        {k: v.tolist() for k, v in obs.items()},
                        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                    assert observation_sha == record['observation_sha256']
                    maps, vector = tensor_observations(obs)
                    assert torch.equal(maps[0], datasets['fresh']['maps'][i])
                    assert torch.equal(vector[0], datasets['fresh']['vectors'][i])
                    np.testing.assert_array_equal(action_mask(env), record['mask'])
                    gains = [copy.deepcopy(env).step(action)[4][
                        'reward_components']['freshness_gain']
                             for action in range(125)]
                    np.testing.assert_allclose(gains, record['gains'], rtol=0, atol=1e-13)
                    fresh_count += 1
                    reward_count += 125
                if step == 96:
                    break
                observation, _, terminal, truncated, _ = env.step(policy.act(env, observation))
                if terminal or truncated:
                    missing.extend({'scenario_seed': world, 'collector': collector, 'step': s,
                        'reason': 'collector terminated before snapshot'}
                        for s in launch['snapshot_steps'] if s > step)
                    break
    complete = json.loads((run/'completion.json').read_text())
    assert fresh_count == len(index['fresh']) == complete['observed_fresh_states']
    assert missing == complete['missing_fresh_snapshots']
    result = {'status': 'PASS', 'source_hashes_verified': len(launch['source_sha256']),
              'dataset_files_verified': len(datasets), 'checkpoint_files_verified': 18,
              'transform_states_checked': sum(len(v['maps']) for v in datasets.values()),
              'field_formula_max_tolerance': 1e-7, 'translation_cropping_padding_exact': True,
              'fresh_natural_states_replayed': fresh_count,
              'original_step_inspection_rewards_checked': reward_count,
              'raw_score_vectors_exactly_reproduced': vectors_checked,
              'current_final_tensors_and_6000_losses_match_parent': True,
              'matched_initial_tensors_size_and_sampling_verified': True,
              'supervised_checkpoints_rejected_by_rl_loaders': True,
              'target_normalisation_verified': True}
    (archive/'replay-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
