"""Controlled input/ranking diagnostics, independent of fitting and mission evaluation.

One-step inspection gain is a reference for input responsiveness, not an oracle
for the optimal long-term policy. Counterfactual pairs are chosen without model
scores. Ranking changes are measured on the shared feasible-action set.
"""

import copy

import numpy as np
import torch

from wild_drone.environment import MOVES
from wild_drone.learning import tensor_observations
from wild_drone.masked_learning import action_mask


def network_scores(policy, env):
    """Read all Q values from the current observation without selecting an action."""
    if policy.algorithm != 'dqn' or policy.env_config != env.config:
        raise ValueError('ranking probe requires a matching DQN checkpoint environment')
    with torch.no_grad():
        q, _ = policy.model(*tensor_observations(
            env._observation(), spatial_encoding=policy.model.spatial_encoding,
            device=policy.device))
    result = q.detach().cpu().numpy().reshape(-1)
    if result.shape != (125,) or not np.isfinite(result).all():
        raise ValueError('model scores must be a finite vector of length 125')
    return result


def transition_coverage(env):
    """Exact post-movement/live-drone footprints from isolated one-step clones.

    Fire randomness is sampled only by each clone. Coverage and the inspection
    reward are determined before fire evolution; the real state/RNG is untouched.
    """
    if env._done:
        raise ValueError('ranking probes require an active mission state')
    masks = []
    for action in range(125):
        clone = copy.deepcopy(env)
        clone.step(action)
        masks.append(clone._inspection_mask())
    return np.stack(masks)


def gain_scores(env, coverage):
    """Original freshness_gain reward component, including the next age increment."""
    weights = env.priority_weights()
    urgency = weights*np.minimum((env.ages+1)/env.config.inspection_deadline, 1)
    return (coverage*urgency).sum(axis=(1, 2))/weights.sum()*env.config.freshness_reward


def best_actions(scores, ids):
    values = np.asarray(scores)[ids]
    return np.asarray(ids)[np.isclose(values, values.max(), rtol=0, atol=1e-12)]


def ranking_metrics(scores, gains, ids):
    """First argmax, tied reference best set, regret relative to gain range."""
    ids = np.asarray(ids)
    selected = int(ids[np.argmax(np.asarray(scores)[ids])])
    best, worst = float(np.max(gains[ids])), float(np.min(gains[ids]))
    regret = best-float(gains[selected])
    span = best-worst
    return {'selected_action': selected, 'selected_gain': float(gains[selected]),
            'feasible_actions': len(ids), 'reference_best_count': len(best_actions(gains, ids)),
            'gain_range': span, 'gain_regret': regret,
            'normalised_gain_regret': regret/span if span > 1e-12 else None,
            'one_step_optimal': bool(selected in best_actions(gains, ids)),
            'random_best_probability': len(best_actions(gains, ids))/len(ids),
            'random_expected_normalised_regret':
            (best-float(np.mean(gains[ids])))/span if span > 1e-12 else None}


def pair_metrics(before, after, gain_before, gain_after, mask_before, mask_after):
    """Distinguish shared value offsets, order changes and mask-driven choice."""
    arrays = [np.asarray(x, dtype=float) for x in (before, after, gain_before, gain_after)]
    if any(x.shape != (125,) or not np.isfinite(x).all() for x in arrays):
        raise ValueError('scores and gains must be finite vectors of length 125')
    before, after, gain_before, gain_after = arrays
    ids = np.flatnonzero(mask_before & mask_after)
    if len(ids) < 2:
        raise ValueError('at least two shared feasible actions are required')
    a, b = before[ids], after[ids]
    first, second = np.triu_indices(len(ids), k=1)
    order_a, order_b = np.sign(a[first]-a[second]), np.sign(b[first]-b[second])
    strict = (order_a != 0) & (order_b != 0)
    ma = ranking_metrics(before, gain_before, ids)
    mb = ranking_metrics(after, gain_after, ids)
    deployment_before = int(np.flatnonzero(mask_before)[np.argmax(before[mask_before])])
    deployment_after = int(np.flatnonzero(mask_after)[np.argmax(after[mask_after])])
    return {'common_actions': len(ids), 'mask_changed': bool(np.any(mask_before != mask_after)),
            'score_delta_max': float(np.max(np.abs(b-a))),
            'centred_score_delta_max': float(np.max(np.abs((b-b.mean())-(a-a.mean())))),
            'order_change_fraction': float(np.mean(order_a != order_b)),
            'strict_flip_fraction': float(np.mean(order_a[strict] != order_b[strict]))
            if strict.any() else None, 'strict_comparable_pairs': int(strict.sum()),
            'common_top_before': ma['selected_action'], 'common_top_after': mb['selected_action'],
            'common_top_changed': ma['selected_action'] != mb['selected_action'],
            'deployment_top_before': deployment_before, 'deployment_top_after': deployment_after,
            'deployment_top_changed': deployment_before != deployment_after,
            **{f'before_{k}': v for k, v in ma.items() if k != 'selected_action'},
            **{f'after_{k}': v for k, v in mb.items() if k != 'selected_action'}}


def urgency_pair(env, coverage):
    """Fresh background, alternating stale regions exclusive to two proposals.

    Search at most 64 action pairs by descending product of their exclusive
    priority-weighted coverage. Balance cell counts within each priority-weight
    stratum so both age histograms and weighted global freshness are identical.
    Accept only disjoint reference best-action sets.
    This is an artificial urgency stress test, not a naturally visited state.
    """
    ids = np.flatnonzero(action_mask(env))
    if len(ids) < 2:
        return None
    flattened = coverage[ids].reshape(len(ids), -1).astype(float)
    weights = env.priority_weights().ravel()
    overlap = (flattened*weights) @ flattened.T
    exclusive = (flattened @ weights)[:, None]-overlap
    strength = exclusive*exclusive.T
    first, second = np.triu_indices(len(ids), k=1)
    order = np.argsort(-strength[first, second], kind='stable')[:64]
    for pair in order:
        i, j = int(ids[first[pair]]), int(ids[second[pair]])
        a_cells, b_cells = coverage[i] & ~coverage[j], coverage[j] & ~coverage[i]
        if not a_cells.any() or not b_cells.any():
            continue
        balanced_a, balanced_b = np.zeros_like(a_cells), np.zeros_like(b_cells)
        for weight in np.intersect1d(weights[a_cells.ravel()], weights[b_cells.ravel()]):
            ai = np.flatnonzero(a_cells.ravel() & (weights == weight))
            bi = np.flatnonzero(b_cells.ravel() & (weights == weight))
            count = min(len(ai), len(bi))
            balanced_a.flat[ai[:count]] = True
            balanced_b.flat[bi[:count]] = True
        a_cells, b_cells = balanced_a, balanced_b
        if not a_cells.any():
            continue
        a, b = copy.deepcopy(env), copy.deepcopy(env)
        a.ages[:] = 0
        b.ages[:] = 0
        a.ages[a_cells] = env.config.inspection_deadline
        b.ages[b_cells] = env.config.inspection_deadline
        ga, gb = gain_scores(a, coverage), gain_scores(b, coverage)
        if set(best_actions(ga, ids)) & set(best_actions(gb, ids)):
            continue
        return a, b, {'kind': 'urgency', 'source_action_pair': [i, j],
                      'stale_cell_count': int(a_cells.sum()),
                      'stale_priority_weight': float(weights[a_cells.ravel()].sum()),
                      'a_only_cells': np.flatnonzero(a_cells).tolist(),
                      'b_only_cells': np.flatnonzero(b_cells).tolist()}
    return None


def position_pair(env, coverage):
    """Move one live drone by one cell; all other state fields remain fixed.

    Preserve bounds, current charger ownership, occupancy and a three-move
    return margin. Choose the largest centred reference-gain change among
    variants with >=2 shared feasible actions and disjoint best-action sets.
    These are counterfactual states, not extra rollout transitions.
    """
    original_mask, original_gain = action_mask(env), gain_scores(env, coverage)
    best, strength = None, -1.0
    for drone in range(3):
        if env.lost[drone] or env.charger_owner == drone:
            continue
        for move in MOVES[1:]:
            destination = env.positions[drone]+move
            if np.any(destination < 0) or np.any(destination >= env.config.grid_size):
                continue
            at_base = np.array_equal(destination, env.base)
            occupied = any(not env.lost[j] and np.array_equal(destination, env.positions[j])
                           for j in range(3) if j != drone)
            distance = int(np.abs(destination-env.base).sum())
            if (occupied and not at_base) or (not at_base and env.batteries[drone] <
                                             (distance+3)*env.config.move_cost):
                continue
            after = copy.deepcopy(env)
            after.positions[drone] = destination
            common = np.flatnonzero(original_mask & action_mask(after))
            if len(common) < 2:
                continue
            changed_coverage = transition_coverage(after)
            changed_gain = gain_scores(after, changed_coverage)
            if set(best_actions(original_gain, common)) & set(best_actions(changed_gain, common)):
                continue
            delta = changed_gain[common]-original_gain[common]
            candidate_strength = float(np.mean(np.abs(delta-delta.mean())))
            if candidate_strength > strength:
                strength = candidate_strength
                best = after, changed_coverage, {
                    'kind': 'position', 'drone': drone,
                    'before': env.positions[drone].tolist(), 'after': destination.tolist()}
    return best
