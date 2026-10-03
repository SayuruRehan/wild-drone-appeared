# Behavior diagnosis and exploration decay experiment

This phase follows the six-fit lower-rate target-rule screen. An exploratory trace of
its 12 exports on validation scenarios 5000–5004 found repeated boundary commands and
few critical return moves. Greedy and sweep avoided negative return-energy margins;
the learned policies did not. These are behavior descriptions, not a proven causal
explanation of every learning failure.

The necessary return-energy bound is Manhattan distance to base times movement cost.
A live drone away from base with less battery cannot reach base under the current
kernel. A nonnegative margin within three movement costs is called critical here,
matching the conventional controllers' buffer. This lower bound is not a safety
guarantee: collisions and hovering can consume extra energy. Parked/lost drones are
excluded. Count actual return moves as well as commands; distinguish wall commands
from collision blocks. Record first infeasibility and loss events, including the
terminal transition, without intervening in actions or consuming the environment RNG.

Before viewing new fit results, fix a single-variable experiment: standard DQN,
learning rate 0.00003, 200 episodes, three seeds 0/1/2, validation scenarios 5000–5004
every 20 episodes, unchanged development environment/network/replay/target/rewards.
Train fresh contemporary controls with epsilon decay 10,000 steps and a second arm
with decay 40,000 steps. All other settings remain equal. There are six fits; do not
extend their budget based on interim scores. Both use epsilon start 1 and end 0.05.

Hypothesis: the existing schedule reaches only 5% random selection around the middle
of this budget; a slower decay may improve late replay action coverage and learned
return decisions. Broader action coverage alone is not coordinated behavior. Examine
returns, freshness, losses, blocked/wall commands, critical actual return moves,
checkpoint Q bounds, and final replay coverage. Expected random-selection counts are
computed from the schedule, not mislabeled as observed branch counts.

Preserve the previous study; its fits are historical context, not extra replicates.
Report final and selected policies separately and aggregate scenarios within each fit
before calculating sample SD over fits. Selection is biased validation evidence.
Final test scenarios remain unused. No actor/reward/architecture/shield change belongs
to this experiment. Review diagnostics, raw records and conclusions before merging
the `sr` → `master` PR after CI.
