"""Verify archived actions/metrics and export 39 browser-readable mission replays."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.masked_learning import action_mask
from wild_drone.webapp.state import (
    initial_packet,
    mission_summary,
    observation_hash,
    snapshot,
)

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT/'docs/experiments/rl-main-2026-10-11'
WORLDS = (22000, 22009, 22019)
ALGORITHMS = ('dqn', 'reinforce', 'a2c')
METRICS = ('return', 'weighted_freshness', 'overdue_fraction', 'energy_used', 'total_losses',
           'safe_returns', 'charge_completions', 'collision_attempts', 'steps',
           'fleet_failure', 'intact_fleet', 'decision_ms')


def read(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def numeric_rows(path):
    result = []
    for row in read(path):
        converted = {}
        for key, value in row.items():
            try:
                converted[key] = float(value)
            except ValueError:
                converted[key] = value
        result.append(converted)
    return result


def equal(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f'{label}: {actual} != {expected}')


def replay(config, episode, trace, scores=None, masks=None):
    env = WildfireFleetEnv(config)
    obs, _ = env.reset(seed=int(episode['scenario_seed']))
    frames = [snapshot(env)]
    for i, row in enumerate(trace):
        if env._done or int(row['step']) != i+1 or observation_hash(obs) != row['pre_sha256']:
            raise ValueError('Pre-transition identity mismatch')
        selected = int(row['action'])
        mask = action_mask(env)
        if str(bool(mask[selected])) != row['selected_mask_feasible']:
            raise ValueError('Recorded feasibility mismatch')
        decision = None
        if scores is not None:
            if not np.isfinite(scores[i]).all() or not np.array_equal(mask, masks[i]):
                raise ValueError('Archived score/mask mismatch')
            if selected != int(np.argmax(np.where(mask, scores[i], -np.inf))):
                raise ValueError('Archived masked argmax mismatch')
            decision = {'scores': scores[i].tolist(), 'allowed': np.flatnonzero(mask).tolist(),
                        'kind': 'q_values' if episode['policy'] == 'dqn' else 'logits'}
        before = {'positions': env.positions.copy(), 'batteries': env.batteries.copy(),
                  'lost': env.lost.copy()}
        obs, reward, term, trunc, info = env.step(selected)
        if observation_hash(obs) != row['post_sha256']:
            raise ValueError('Post-transition identity mismatch')
        equal(reward, row['reward'], 'reward')
        if str(term) != row['terminated'] or str(trunc) != row['truncated']:
            raise ValueError('Termination mismatch')
        for key in ('weighted_freshness', 'overdue_fraction', 'energy_used', 'new_losses',
                    'total_losses', 'safe_returns', 'charge_completions', 'collision_attempts'):
            equal(info[key], row[key], key)
        for key, value in info['reward_components'].items():
            equal(value, row[f'reward_{key}'], key)
        frames.append(snapshot(env, info, action=selected, previous=before, decision=decision))
    if not env._done:
        raise ValueError('Incomplete archived mission')
    summary = mission_summary(env, frames)
    for key, value in summary.items():
        equal(value, episode[key], key)
    packet = initial_packet(env, policy=episode['policy'], world=int(episode['scenario_seed']),
                            training_seed=int(episode['training_seed'])
                            if episode['training_seed'] else None, kind='recorded')
    return packet | {'frames': frames, 'summary': summary, 'verified': True}


def recompute(episodes):
    fitted = defaultdict(list)
    for row in episodes:
        fitted[row['policy'], row['training_seed']].append(row)
    policies = defaultdict(list)
    for (policy, _), rows in fitted.items():
        if policy in ALGORITHMS:
            policies[policy].append({m: statistics.mean(float(r[m]) for r in rows)
                                     for m in METRICS})
        else:
            policies[policy].extend(rows)
    result = []
    for policy, rows in policies.items():
        packet = {'policy': policy, 'unit': 'seed_means' if policy in ALGORITHMS else 'worlds',
                  'units': len(rows)}
        for metric in METRICS:
            values = [float(r[metric]) for r in rows]
            packet[metric] = statistics.mean(values)
            packet[metric+'_sd'] = statistics.stdev(values)
        result.append(packet)
    for row in read(ARCHIVE/'summary.csv'):
        packet = next(r for r in result if r['policy'] == row['policy'])
        equal(packet[row['metric']], row['mean'], 'mean')
        equal(packet[row['metric']+'_sd'], row['sample_sd'], 'sample SD')
    return result


def export(out):
    out.mkdir(parents=True, exist_ok=True)
    checksums = json.loads((ARCHIVE/'checksums.json').read_text())
    inputs = ['episodes.csv', 'summary.csv', 'steps.csv.gz'] + [
        f'{a}-{s}-scores.npz' for a in ALGORITHMS for s in range(3)]
    for relative in inputs:
        expected = checksums[relative]
        data = (ARCHIVE/relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected['sha256']:
            raise ValueError(f'Archive bytes differ: {relative}')
    frozen = json.loads((ROOT/'configs/rl-main-frozen.json').read_text())
    config = EnvConfig(**frozen['environment'])
    episodes = read(ARCHIVE/'episodes.csv')
    groups, offsets = defaultdict(list), defaultdict(int)
    with gzip.open(ARCHIVE/'steps.csv.gz', 'rt') as stream:
        for row in csv.DictReader(stream):
            groups[row['policy'], row['training_seed'], int(row['scenario_seed'])].append(row)
    vectors = {(a, str(s)): np.load(ARCHIVE/f'{a}-{s}-scores.npz', allow_pickle=False)
               for a in ALGORITHMS for s in range(3)}
    catalogue, verified_steps = [], 0
    for episode in episodes:
        policy, seed, world = episode['policy'], episode['training_seed'], int(
            episode['scenario_seed'])
        trace = groups[policy, seed, world]
        start = offsets[policy, seed]
        offsets[policy, seed] += len(trace)
        if world not in WORLDS:
            continue
        scores = masks = None
        if policy in ALGORITHMS:
            scores = vectors[policy, seed]['scores'][start:start+len(trace)]
            masks = vectors[policy, seed]['masks'][start:start+len(trace)]
        packet = replay(config, episode, trace, scores, masks)
        identity = f'{policy}-{seed or "baseline"}-{world}'
        packet['id'] = identity
        packet['source_revision'] = json.loads((ARCHIVE/'launch.json').read_text())[
            'source_revision']
        payload = gzip.compress(json.dumps(packet, separators=(',', ':'), allow_nan=False).encode(),
                                mtime=0)
        (out/f'{identity}.json.gz').write_bytes(payload)
        verified_steps += len(trace)
        catalogue.append({'id': identity, 'policy': policy, 'seed': int(seed) if seed else None,
                          'world': world, 'file': f'{identity}.json.gz',
                          'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
                          'summary': packet['summary']})
        print(f'Verified {identity}: {len(trace)} transitions', flush=True)
    if len(catalogue) != 39:
        raise ValueError('Exactly 39 selected missions required')
    training = []
    for a in ALGORITHMS:
        for s in range(3):
            rows = numeric_rows(ARCHIVE/f'fits/{a}/seed-{s}/training.csv')
            training.append({'policy': a, 'seed': s, 'rows': [
                {k: r[k] for k in ('episode', 'return', 'weighted_freshness', 'total_losses')}
                for r in rows]})
    follow = ROOT/'experiments/lithma/results'
    data = {'format_version': 1, 'study': 'Frozen 40×40 main study · 11 October 2026',
            'source_revision': packet['source_revision'], 'config': config.to_dict(),
            'worlds': WORLDS, 'missions': catalogue, 'summary': recompute(episodes),
            'training': training, 'completion': json.loads((ARCHIVE/'completion.json').read_text()),
            'followups': {name: numeric_rows(follow/file) for name, file in [
                ('planner', 'planner_vs_frozen_table.csv'),
                ('sensitivity', 'sensitivity_summary.csv'), ('rates', 'lr_sweep_summary.csv')]},
            'verification': {'missions': len(catalogue), 'steps': verified_steps,
                             'hashes_rewards_masks_summaries': True},
            'sources': {str((ARCHIVE/f).relative_to(ROOT)): checksums[f] for f in inputs}}
    for file in ('planner_vs_frozen_table.csv', 'sensitivity_summary.csv', 'lr_sweep_summary.csv'):
        source = follow/file
        data['sources'][str(source.relative_to(ROOT))] = {
            'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'bytes': source.stat().st_size}
    (out/'evidence.json').write_text(json.dumps(data, separators=(',', ':'), allow_nan=False)+'\n')
    print(f'Exported 39 verified missions, {verified_steps} transitions and recomputed summaries.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'web/public/data')
    export(parser.parse_args().out)
