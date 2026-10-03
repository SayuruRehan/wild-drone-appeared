"""Validation rollout diagnostics; these bounds do not establish value calibration."""

import math

import torch

from wild_drone.learning import tensor_observations


def discounted_upper_bound(remaining, gamma, reward_upper):
    if gamma == 1:
        return reward_upper*remaining
    return reward_upper*(1-gamma**remaining)/(1-gamma)


class DQNDiagnosticPolicy:
    """Use the saved argmax policy while observing Q values at visited states."""

    def __init__(self, policy):
        if policy.algorithm != 'dqn':
            raise ValueError('Q diagnostics require a DQN checkpoint')
        self.policy = policy
        self.max_q, self.max_excess = -math.inf, 0.0
        self.decisions, self.violations = 0, 0

    def act(self, env, observation):
        policy = self.policy
        if env.config != policy.env_config:
            raise ValueError('checkpoint environment differs from evaluation config')
        with torch.no_grad():
            q, _ = policy.model(*tensor_observations(
                observation, spatial_encoding=policy.model.spatial_encoding, device=policy.device))
        if not torch.isfinite(q).all():
            raise FloatingPointError('nonfinite Q value during validation')
        highest = float(q.max())
        # Freshness gain <= 1 and all other reward terms are nonpositive.
        bound = discounted_upper_bound(env.config.horizon-env.elapsed,
                                       policy.train_config.gamma, env.config.freshness_reward)
        excess = highest-bound
        self.decisions += 1
        self.violations += int(excess > 1e-5)
        self.max_q = max(self.max_q, highest)
        self.max_excess = max(self.max_excess, excess)
        return int(q.argmax(dim=1).item())
