# Checkpoint integrity and a bounded DQN comparison

The T4 study revealed excessive Q values; one Drive policy export also lagged its
authoritative state. This phase changes learning/runtime checks, retaining the approved
environment, reward, observation, network and joint-action space.

1. Audit final and selected policy metadata, episode boundaries and every model tensor
   against `training_state.pt`. Record SHA-256 hashes in `checkpoint_manifest.json`.
   Audit on training completion and again before notebook archiving. Provide a read-only
   command for existing runs. Explicit same-budget resume regenerates interrupted exports.
2. Add optional Double DQN: choose the next action with the online network, evaluate it
   with the target network, and retain terminal masking. Standard DQN remains the default.
   Existing states without the flag mean standard DQN; resume cannot switch variants.
3. Before examining results, fix this local validation budget: 200 episodes, training
   seeds 0/1/2, validation scenarios 5000–5004, validation every 20 episodes, CPU, and
   learning rate 0.00003 for both standard and Double DQN. All other settings come from
   `configs/development.json`. This repeats the lower-rate candidate and isolates the
   target-rule comparison. The original 0.0003 screen is historical context, not a
   matched control for these new fits. Do not expand the budget based on interim scores.
4. Report final and selected returns, freshness, losses and collisions per fit; inspect
   maximum Q values along validation rollouts and violations of the finite-horizon upper
   return bound. A lack of violations does not establish calibrated values or convergence.
   Selection over ten validation boundaries is biased. Final test scenarios remain unused.
5. Preserve raw CSV/JSON logs and code/runtime provenance, run the full suite and lint,
   obtain independent review, publish an `sr` → `master` PR and merge after successful CI.

Double DQN follows [van Hasselt, Guez and Silver](https://arxiv.org/abs/1509.06461).
Its action-selection/evaluation split addresses a source of overestimation; it does not
guarantee stable training or good drone coordination.
