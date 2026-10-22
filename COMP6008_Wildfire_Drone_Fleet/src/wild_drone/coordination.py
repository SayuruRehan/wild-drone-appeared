"""Training-only, state-based potential for inspection coordination.

Ng, Harada and Russell (1999): add gamma*Phi(next)-Phi(current), using the
learner's discount and zero potential at every terminal state. The discounted
sum then telescopes to -Phi(initial), independent of the trajectory. This does
not guarantee improvement with an approximate learner or in undiscounted metrics.
"""

import numpy as np

SIGNAL_RULE = 'freshness-proximity-potential-v1'


def coordination_potential(env, *, terminal=False):
    """Bounded [0, 1] priority-weighted freshness plus anticipatory coverage.

    Each cell contributes freshness + .5*staleness*exp(-distance/3). Distance is
    the minimum Manhattan travel to any live drone's square sensor footprint.
    Taking the minimum gives no duplicate credit for co-located drones. Only
    the current fully observed state is read; no simulation or RNG is used.
    """
    if terminal:
        return 0.0
    c = env.config
    stale = np.minimum(env.ages/c.inspection_deadline, 1)
    proximity = np.zeros_like(stale)
    positions = env.positions[~env.lost]
    if len(positions):
        cells = np.indices(stale.shape).transpose(1, 2, 0)
        distance = np.maximum(np.abs(cells[None]-positions[:, None, None])-
                              c.sensor_radius, 0).sum(axis=-1).min(axis=0)
        proximity = np.exp(-distance/3)
    weights = env.priority_weights()
    return float((weights*(1-stale+.5*stale*proximity)).sum()/weights.sum())
