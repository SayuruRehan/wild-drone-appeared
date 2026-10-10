"""Fixed supervised inspection scorers in complete fresh 20x20 missions."""

import argparse
import csv
import gzip
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from wild_drone.baselines import GreedyPolicy, SweepPolicy
from wild_drone.centered_inspection import FORMAT, load_centered_model
from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.experiments import write_rows
from wild_drone.inspection_targets import validate_worlds
from wild_drone.learning import tensor_observations
from wild_drone.masked_learning import MaskedRandomPolicy, action_mask
from wild_drone.training_io import atomic_json_save

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'docs'/'experiments'/'training-coverage-2026-10-10'
WORLDS = tuple(range(6600, 6620))
METRICS = ('return', 'weighted_freshness', 'overdue_fraction', 'energy_used',
           'total_losses', 'safe_returns', 'charge_completions', 'collision_attempts',
           'steps', 'fleet_failure', 'intact_fleet', 'decision_ms')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observation_digest(observation):
    return hashlib.sha256(json.dumps({k: v.tolist() for k, v in observation.items()},
                                    sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class ScorePolicy:
    """Feasible first-index argmax of frozen inspection scores, without lookahead."""

    def __init__(self, config, *, model=None, scores=None):
        if (model is None) == (scores is None):
            raise ValueError('provide exactly one model or fixed score vector')
        self.config, self.model, self.scores = config, model, scores

    def act(self, env, observation):
        if env.config != self.config:
            raise ValueError('matching environment configuration required')
        if self.model is None:
            scores = np.asarray(self.scores, dtype=np.float32)
        else:
            with torch.no_grad():
                scores = self.model(*tensor_observations(observation))[0][0].numpy()
        if scores.shape != (125,) or not np.isfinite(scores).all():
            raise ValueError('finite 125-action inspection scores required')
        self.last_scores, self.last_mask = scores.copy(), action_mask(env)
        return int(np.argmax(np.where(self.last_mask, scores, -np.inf)))


def rollout(config, policy, world, *, name, training_seed=None):
    env = WildfireFleetEnv(config)
    observation, _ = env.reset(seed=world)
    trace, predictions, masks, durations = [], [], [], []
    total = 0.0
    while True:
        before = observation_digest(observation)
        started = time.perf_counter()
        action = policy.act(env, observation)
        durations.append((time.perf_counter()-started)*1000)
        mask = action_mask(env)
        if isinstance(policy, ScorePolicy):
            if not np.array_equal(mask, policy.last_mask):
                raise ValueError('policy changed the feasible state')
            predictions.append(policy.last_scores.copy())
            masks.append(mask.copy())
        observation, reward, terminated, truncated, info = env.step(action)
        total += reward
        trace.append({'policy': name, 'training_seed': training_seed, 'scenario_seed': world,
                      'step': env.elapsed, 'action': action,
                      'selected_mask_feasible': bool(mask[action]),
                      'pre_sha256': before, 'post_sha256': observation_digest(observation),
                      'reward': reward, 'weighted_freshness': info['weighted_freshness'],
                      'overdue_fraction': info['overdue_fraction'],
                      'energy_used': info['energy_used'], 'new_losses': info['new_losses'],
                      'total_losses': int(env.lost.sum()), 'safe_returns': info['safe_returns'],
                      'charge_completions': info['charge_completions'],
                      'collision_attempts': info['collision_attempts'],
                      'terminated': terminated, 'truncated': truncated,
                      **{f'reward_{k}': v for k, v in info['reward_components'].items()}})
        if terminated or truncated:
            break
    missing = config.horizon-len(trace)
    episode = {'policy': name, 'training_seed': training_seed, 'scenario_seed': world,
               'grid_size': config.grid_size, 'horizon': config.horizon, 'steps': env.elapsed,
               'return': total,
               'weighted_freshness': sum(r['weighted_freshness'] for r in trace)/config.horizon,
               'overdue_fraction': (sum(r['overdue_fraction'] for r in trace)+missing)/
               config.horizon, 'energy_used': env.totals['energy_used'],
               'total_losses': int(env.lost.sum()), 'safe_returns': env.totals['safe_returns'],
               'charge_completions': env.totals['charge_completions'],
               'collision_attempts': env.totals['collision_attempts'],
               'decision_ms': float(np.mean(durations)), 'fleet_failure': int(env.lost.all()),
               'intact_fleet': int(not env.lost.any())}
    vectors = {'scores': np.stack(predictions), 'masks': np.stack(masks)} if predictions else None
    return episode, trace, vectors


def load_models(run, *, archive=None):
    archive = PARENT if archive is None else archive
    launch = json.loads((archive/'launch.json').read_text())
    config = EnvConfig(**launch['env_config'])
    for path, sha in launch['source_sha256'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('frozen parent source identity differs')
    index = json.loads((archive/'checkpoints.json').read_text())
    identities = {}
    for corpus in ('small', 'expanded'):
        for seed in (0, 1, 2):
            relative = f'{corpus}/seed-{seed}/final.pt'
            path = run/relative
            identity = index[f'{corpus}/{seed}']['checkpoints']['final.pt']
            if digest(path) != identity['sha256'] or path.stat().st_size != identity['bytes']:
                raise ValueError('frozen checkpoint identity differs')
            saved = torch.load(path, weights_only=True, map_location='cpu')
            if (saved.get('format') != FORMAT or saved.get('seed') != seed or
                    saved.get('completed_updates') != 2000 or saved.get('arm') != 'centered' or
                    saved.get('env_config') != config.to_dict()):
                raise ValueError('frozen checkpoint metadata differs')
            identities[f'{corpus}/{seed}'] = {'relative_path': relative, **identity}
    return config, identities


def per_fit(rows):
    result = []
    for name, seed in dict.fromkeys((r['policy'], r['training_seed']) for r in rows):
        group = [r for r in rows if r['policy'] == name and r['training_seed'] == seed]
        worlds = [r['scenario_seed'] for r in group]
        if len(set(worlds)) != len(worlds):
            raise ValueError('duplicate mission world')
        result.append({'policy': name, 'training_seed': seed, 'worlds': len(worlds),
                       **{key: float(np.mean([r[key] for r in group])) for key in METRICS}})
    return result


def decisions(rows):
    results = []
    greedy = [r for r in rows if r['policy'] == 'greedy']
    worlds = {r['scenario_seed'] for r in greedy}
    if not worlds or len(worlds) != len(greedy):
        raise ValueError('distinct Greedy comparison worlds required')
    for seed in (0, 1, 2):
        groups = {name: [r for r in rows if r['policy'] == name and r['training_seed'] == seed]
                  for name in ('small', 'expanded')}
        if any(len(group) != len(worlds) or {r['scenario_seed'] for r in group} != worlds
               for group in groups.values()):
            raise ValueError('matched mission worlds required')
        means = {name: {key: float(np.mean([r[key] for r in group])) for key in
                        ('weighted_freshness', 'overdue_fraction', 'total_losses', 'fleet_failure')}
                 for name, group in (groups | {'greedy': greedy}).items()}
        small, expanded, reference = (means[k] for k in ('small', 'expanded', 'greedy'))
        effect = {'freshness_improves_ge_001': expanded['weighted_freshness']-
                  small['weighted_freshness'] >= .01,
                  'overdue_not_higher': expanded['overdue_fraction'] <= small['overdue_fraction'],
                  'losses_not_higher': expanded['total_losses'] <= small['total_losses'],
                  'fleet_failures_not_higher': expanded['fleet_failure'] <= small['fleet_failure']}
        capability = {'freshness_ge_90pct_greedy': expanded['weighted_freshness'] >=
                      .9*reference['weighted_freshness'],
                      'overdue_le_greedy_plus_005': expanded['overdue_fraction'] <=
                      reference['overdue_fraction']+.05,
                      'losses_le_greedy_plus_01': expanded['total_losses'] <=
                      reference['total_losses']+.1,
                      'fleet_failures_le_greedy': expanded['fleet_failure'] <=
                      reference['fleet_failure']}
        results.append({'seed': seed, 'means': means, 'data_effect_checks': effect,
                        'data_effect_passed': all(effect.values()),
                        'greedy_capability_checks': capability,
                        'greedy_capability_passed': all(capability.values())})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch('[0-9a-f]{40}', args.source_revision):
        raise ValueError('full published source revision required')
    config, identities = load_models(args.parent_root)
    validate_worlds(tuple(range(200)), WORLDS)
    validate_worlds(tuple(range(6200, 6216))+tuple(range(6400, 6416)), WORLDS)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source = sorted((ROOT/'src').rglob('*.py')) + [Path(__file__).resolve(),
             ROOT/'docs'/'mission-validation-protocol.md']
    controls = json.loads((PARENT/'action-mean-controls.json').read_text())
    parent_checks = json.loads((PARENT/'checksums.json').read_text())
    if digest(PARENT/'action-mean-controls.json') != parent_checks[
            'action-mean-controls.json']['sha256']:
        raise ValueError('fixed training-action-mean controls differ')
    atomic_json_save({'started_utc': datetime.now(timezone.utc).isoformat(),
                     'source_revision': args.source_revision,
                     'source_revision_kind': 'operator-verified published source snapshot',
                     'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in source},
                     'env_config': config.to_dict(), 'worlds': list(WORLDS),
                     'parent_source_revision': 'a2a6b4e7fa68ff71d735542d3e0633eb7a09886a',
                     'training_seeds': [0, 1, 2], 'training_updates': 0,
                     'checkpoints': identities, 'cpu_threads': 1,
                     'controls_sha256': digest(PARENT/'action-mean-controls.json')},
                    args.out/'launch.json')
    started, episodes, traces = time.perf_counter(), [], []
    conditions = [(corpus, seed) for corpus in ('small', 'expanded') for seed in (0, 1, 2)] + \
                 [('small_action_mean', None), ('expanded_action_mean', None),
                  ('greedy', None), ('sweep', None), ('masked_random', None)]
    for name, seed in conditions:
        model = None if seed is None else load_centered_model(
            args.parent_root/identities[f'{name}/{seed}']['relative_path'], config, 'centered')
        score_blocks, mask_blocks = [], []
        for world in WORLDS:
            if model is not None:
                policy = ScorePolicy(config, model=model)
            elif name.endswith('_action_mean'):
                policy = ScorePolicy(config, scores=controls[name.split('_')[0]]['scores'])
            else:
                policy = {'greedy': GreedyPolicy, 'sweep': SweepPolicy,
                          'masked_random': lambda: MaskedRandomPolicy(world)}[name]()
            episode, trace, vectors = rollout(config, policy, world, name=name, training_seed=seed)
            episodes.append(episode)
            traces.extend(trace)
            if vectors is not None:
                score_blocks.append(vectors['scores'])
                mask_blocks.append(vectors['masks'])
        if score_blocks:
            np.savez_compressed(args.out/f'{name}-seed-{seed}-scores.npz',
                                scores=np.concatenate(score_blocks),
                                masks=np.concatenate(mask_blocks))
        write_rows(args.out/'episodes.csv', episodes)
        print(f'{name}/seed-{seed}: {len(WORLDS)} complete missions', flush=True)
    with gzip.open(args.out/'steps.csv.gz', 'wt', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(traces[0]))
        writer.writeheader()
        writer.writerows(traces)
    fits = per_fit(episodes)
    write_rows(args.out/'per-fit.csv', fits)
    aggregate = []
    for name in dict.fromkeys(r['policy'] for r in episodes):
        fitted = name in ('small', 'expanded')
        group = [r for r in (fits if fitted else episodes) if r['policy'] == name]
        for key in METRICS:
            values = [r[key] for r in group]
            aggregate.append({'policy': name, 'metric': key,
                              'unit': 'fitted_seed_means' if fitted else 'worlds',
                              'units': len(values), 'mean': float(np.mean(values)),
                              'sample_sd': float(np.std(values, ddof=1))})
    write_rows(args.out/'summary.csv', aggregate)
    by_key = {(r['policy'], r['training_seed'], r['scenario_seed']): r for r in episodes}
    paired = [{'training_seed': seed, 'scenario_seed': world,
               **{key: by_key['expanded', seed, world][key]-by_key['small', seed, world][key]
                  for key in METRICS}} for seed in (0, 1, 2) for world in WORLDS]
    write_rows(args.out/'paired-expanded-minus-small.csv', paired)
    criterion = decisions(episodes)
    completion = {'episodes': len(episodes), 'steps': len(traces),
                  'data_effect_passed': all(r['data_effect_passed'] for r in criterion),
                  'greedy_capability_passed': all(r['greedy_capability_passed'] for r in criterion),
                  'per_seed': criterion, 'elapsed_seconds': time.perf_counter()-started,
                  'rl_updates': 0, 'scale_to_40x40': False}
    atomic_json_save(completion, args.out/'completion.json')
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == '__main__':
    main()
