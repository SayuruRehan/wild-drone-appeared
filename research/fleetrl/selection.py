"""Exact maximisation of a pairwise coordination-graph score over feasible joint actions.

score(a) = sum_i U[i, a_i] + sum_{i<j} P[i, j, a_i, a_j]

The score is evaluated densely over all 5**N joint actions (390,625 for N=8), which is
exact and fast enough for evaluation up to eight drones. No approximate max-sum or
coordinate ascent is involved, so differences between controllers cannot be caused by
an inexact action search.
"""

import itertools

import numpy as np

MAX_EXACT_DRONES = 8


def pairs(n_drones):
    return list(itertools.combinations(range(n_drones), 2))


def dense_scores(unary, pairwise=None):
    """unary (N, 5); pairwise (N, N, 5, 5) or None -> dense scores of shape (5,)*N."""
    k = unary.shape[0]
    if k > MAX_EXACT_DRONES:
        raise ValueError(f'exact selection supports at most {MAX_EXACT_DRONES} drones')
    total = np.zeros((5,)*k, dtype=np.float64)
    for i in range(k):
        shape = [1]*k
        shape[i] = 5
        total = total + unary[i].reshape(shape)
    if pairwise is not None:
        for i, j in pairs(k):
            shape = [1]*k
            shape[i] = shape[j] = 5
            total = total + pairwise[i, j].reshape(shape)
    return total


def select(unary, pairwise, mask):
    """Best feasible joint action (flat index) under the coordination-graph score.
    Ties break toward the lowest index, as numpy argmax does."""
    scores = dense_scores(unary, pairwise).reshape(-1)
    scores = np.where(mask, scores, -np.inf)
    return int(np.argmax(scores))


def torch_dense_q(unary, pair_values, pair_index):
    """Batched dense joint values.

    unary: (B, N, 5); pair_values: (B, P, 5, 5) or None; pair_index: list of (i, j).
    Returns (B, 5**N).
    """
    b, k, _ = unary.shape
    total = unary.new_zeros((b,)+(5,)*k)
    for i in range(k):
        shape = [b]+[1]*k
        shape[1+i] = 5
        total = total + unary[:, i].reshape(shape)
    if pair_values is not None:
        for p, (i, j) in enumerate(pair_index):
            shape = [b]+[1]*k
            shape[1+i] = shape[1+j] = 5
            total = total + pair_values[:, p].reshape(shape)
    return total.reshape(b, -1)


def torch_gather_joint(unary, pair_values, pair_index, commands):
    """Value of given joint actions without the dense table.
    commands: (B, N) long -> (B,)."""
    value = unary.gather(2, commands[:, :, None]).squeeze(2).sum(1)
    if pair_values is not None:
        for p, (i, j) in enumerate(pair_index):
            flat = commands[:, i]*5 + commands[:, j]
            value = value + pair_values[:, p].reshape(-1, 25).gather(1, flat[:, None]).squeeze(1)
    return value


def torch_masked_argmax(q, mask):
    """q, mask: (B, A) -> (values, indices) of the best feasible action."""
    q = q.masked_fill(~mask, float('-inf'))
    return q.max(dim=1)


__all__ = ['MAX_EXACT_DRONES', 'pairs', 'dense_scores', 'select', 'torch_dense_q',
           'torch_gather_joint', 'torch_masked_argmax']
