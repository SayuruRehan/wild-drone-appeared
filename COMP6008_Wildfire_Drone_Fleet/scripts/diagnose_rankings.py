"""Frozen 20x20 position/urgency probes of the two completed coordination-pilot fits."""

import argparse
import copy
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from wild_drone.baselines import make_policy
from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.experiments import write_rows
from wild_drone.learning import FleetNetwork, TrainConfig
from wild_drone.masked_learning import (
    MaskedNeuralPolicy,
    MaskedRandomPolicy,
    action_mask,
    load_masked_policy,
)
from wild_drone.ranking_probe import (
    gain_scores,
    network_scores,
    pair_metrics,
    position_pair,
    ranking_metrics,
    transition_coverage,
    urgency_pair,
)
from wild_drone.training_io import atomic_json_save

ROOT = Path(__file__).resolve().parents[1]
WORLDS = tuple(range(5900, 5908))
STEPS = (16, 80)
NAMES = ('unshaped', 'shaped', 'untrained', 'state_blind', 'inspection')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(root):
    """Check binary identity and the exact completed pilot recipe before creating output."""
    identities = json.loads((root/'checkpoints.json').read_text())
    config = EnvConfig(grid_size=20, horizon=160, battery_capacity=80, recharge_rate=5,
                       sensor_radius=1, inspection_deadline=25, max_fuel=100)
    policies = {}
    for recipe, weight in [('unshaped', 0), ('shaped', 10)]:
        path = root/'raw'/recipe/'seed-0'/'policy.pt'
        record = identities[recipe]['policy.pt']
        if digest(path) != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError(f'{recipe} checkpoint differs from pilot identity')
        policy = load_masked_policy(path)
        expected = TrainConfig(episodes=100, seed=0, learning_rate=.001, double_dqn=True,
                               coordination_shaping_weight=weight, validation_interval=100,
                               validation_seeds=tuple(range(5000, 5005)), progress_interval=25)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if (policy.algorithm != 'dqn' or policy.env_config != config or
                policy.train_config != expected or saved['completed_episodes'] != 100):
            raise ValueError(f'{recipe} checkpoint differs from frozen pilot recipe')
        policies[recipe] = policy
    return config, policies, identities


def mean(values):
    values = [float(v) for v in values if v is not None]
    return float(np.mean(values)) if values else None


def summarize(natural, pairs):
    rows = []
    for kind in ('natural', 'urgency', 'position', 'noop'):
        for name in NAMES:
            selected = [r for r in (natural if kind == 'natural' else pairs)
                        if r['policy'] == name and (kind == 'natural' or r['kind'] == kind)]
            units = selected if kind == 'natural' else [
                {k: r[f'{side}_{k}'] for k in ('normalised_gain_regret', 'one_step_optimal',
                 'random_best_probability', 'random_expected_normalised_regret')}
                for r in selected for side in ('before', 'after')]
            informative = [r for r in units if r['normalised_gain_regret'] is not None]
            rows.append({'kind': kind, 'policy': name, 'cases': len(selected),
                         'informative_states': len(informative),
                         'common_top_change_rate': mean([r['common_top_changed'] for r in selected])
                         if kind != 'natural' else None,
                         'deployment_top_change_rate': mean([r['deployment_top_changed']
                                                           for r in selected])
                         if kind != 'natural' else None,
                         'mean_order_change_fraction': mean([r['order_change_fraction']
                                                            for r in selected])
                         if kind != 'natural' else None,
                         'mean_centred_score_delta_max': mean([r['centred_score_delta_max']
                                                              for r in selected])
                         if kind != 'natural' else None,
                         'one_step_best_rate': mean([r['one_step_optimal'] for r in informative]),
                         'mean_normalised_gain_regret': mean([r['normalised_gain_regret']
                                                            for r in informative]),
                         'random_best_probability': mean([r['random_best_probability']
                                                          for r in informative]),
                         'random_expected_normalised_regret': mean([
                             r['random_expected_normalised_regret'] for r in informative])})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch('[0-9a-f]{40}', args.source_revision):
        raise ValueError('source-revision must be a full published commit SHA')
    config, policies, identities = load_inputs(args.pilot_root)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(101)
    policies['untrained'] = MaskedNeuralPolicy(FleetNetwork(20), 'dqn', config,
                                             train_config=TrainConfig(seed=101))
    fixed_scores = np.random.default_rng(202).normal(size=125)
    paths = sorted((ROOT/'src').rglob('*.py')) + [Path(__file__).resolve(),
            ROOT/'docs'/'ranking-diagnostic-protocol.md']
    manifest = {'started_utc': datetime.now(timezone.utc).isoformat(),
                'source_revision': args.source_revision,
                'source_revision_kind': 'operator-verified published source snapshot',
                'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in paths},
                'env_config': config.to_dict(), 'checkpoint_identities': identities,
                'checkpoint_choice': 'final', 'training_seed': 0,
                'worlds': list(WORLDS), 'snapshot_steps': list(STEPS),
                'collectors': ['masked_random', 'greedy'], 'expected_snapshots': 32,
                'untrained_seed': 101, 'state_blind_seed': 202,
                'cpu_threads': 1, 'training_performed': False}
    atomic_json_save(manifest, args.out/'launch.json')
    natural, pairs, coverage_rows, missing = [], [], [], []
    started = time.perf_counter()
    for world in WORLDS:
        for collector in ('masked_random', 'greedy'):
            env = WildfireFleetEnv(config)
            observation, _ = env.reset(seed=world)
            controller = MaskedRandomPolicy(world) if collector == 'masked_random' else \
                make_policy('greedy')
            for step in range(max(STEPS)+1):
                if step in STEPS:
                    case = f'{world}-{collector}-{step}'
                    common = {'case': case, 'scenario_seed': world,
                              'collector': collector, 'step': step}
                    cov = transition_coverage(env)
                    urgency = urgency_pair(env, cov)
                    position = position_pair(env, cov)
                    states = {'natural': env, 'noop': copy.deepcopy(env)}
                    covers = {'natural': cov, 'noop': cov}
                    details = {}
                    if urgency:
                        states['urgency_a'], states['urgency_b'], details['urgency'] = urgency
                        covers['urgency_a'] = covers['urgency_b'] = cov
                    if position:
                        states['position'], covers['position'], details['position'] = position
                    masks = {name: action_mask(state) for name, state in states.items()}
                    gains = {name: gain_scores(state, covers[name])
                             for name, state in states.items()}
                    metadata = common | {'observation_sha256': hashlib.sha256(json.dumps(
                        {k: v.tolist() for k, v in env._observation().items()},
                        sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                        'positions': env.positions.tolist(), 'batteries': env.batteries.tolist(),
                        'charger_owner': int(env.charger_owner), 'details': details,
                        'masks': {k: v.tolist() for k, v in masks.items()}}
                    folder = args.out/'raw'/case
                    folder.mkdir(parents=True)
                    atomic_json_save(metadata, folder/'state.json')
                    coverage_rows.append(common | {'natural_feasible_actions': int(
                        masks['natural'].sum()), 'urgency_eligible': urgency is not None,
                        'position_eligible': position is not None,
                        'noop_eligible': bool(masks['natural'].sum() >= 2),
                        'urgency_skip_reason': '' if urgency else
                        'fewer than two feasible actions' if masks['natural'].sum() < 2 else
                        'no balanced disjoint optimum in top64 proposal pairs',
                        'position_skip_reason': '' if position else
                        'fewer than two feasible actions' if masks['natural'].sum() < 2 else
                        'no valid one-cell move with shared disjoint optimum'})
                    for name in NAMES:
                        scores = {label: network_scores(policies[name], state) if name in policies
                                  else fixed_scores if name == 'state_blind' else gains[label]
                                  for label, state in states.items()}
                        atomic_json_save({'policy': name, 'scores': {k: v.tolist()
                                         for k, v in scores.items()}}, folder/f'{name}.json')
                        natural.append(common | {'policy': name, **ranking_metrics(
                            scores['natural'], gains['natural'], np.flatnonzero(masks['natural']))})
                        choices = [('noop', 'natural', 'noop')]
                        if urgency:
                            choices.append(('urgency', 'urgency_a', 'urgency_b'))
                        if position:
                            choices.append(('position', 'natural', 'position'))
                        for kind, before, after in choices:
                            if np.count_nonzero(masks[before] & masks[after]) < 2:
                                continue
                            metrics = pair_metrics(scores[before], scores[after], gains[before],
                                                   gains[after], masks[before], masks[after])
                            if kind == 'noop' and metrics['score_delta_max'] != 0:
                                raise ValueError('no-op control changed scores')
                            pairs.append(common | {'policy': name, 'kind': kind, **metrics})
                    print(f'{case}: urgency={urgency is not None}, position={position is not None}',
                          flush=True)
                if step == max(STEPS):
                    break
                action = controller.act(env, observation)
                observation, _, terminal, truncated, _ = env.step(action)
                if terminal or truncated:
                    missing.extend({'scenario_seed': world, 'collector': collector, 'step': s,
                                    'reason': 'collector terminated before snapshot'}
                                   for s in STEPS if s > step)
                    break
    write_rows(args.out/'natural.csv', natural)
    write_rows(args.out/'pairs.csv', pairs)
    write_rows(args.out/'coverage.csv', coverage_rows)
    summary = summarize(natural, pairs)
    write_rows(args.out/'summary.csv', summary)
    atomic_json_save({'observed_snapshots': len(coverage_rows), 'missing_snapshots': missing,
                      'urgency_eligible': sum(r['urgency_eligible'] for r in coverage_rows),
                      'position_eligible': sum(r['position_eligible'] for r in coverage_rows),
                      'elapsed_seconds': time.perf_counter()-started}, args.out/'completion.json')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
