# Starting release implementation plan

Design authority: `docs/design.md`, reflecting the approved fully observable MDP proposal.
Execution: implement directly in this session, test before adding each subsystem, then review
the whole change. The user's instruction to start the approved project supplies execution
authorization; no new topic approval is required for the implementation defaults below.

## Constraints and review focus

Three drones, five primitive actions each, full state, single charger, stochastic fire/wind,
explicit finite mission. Check collision cascades and returning with exactly zero energy;
prevent charger ownership from becoming hidden state. Prevent planner RNG clairvoyance.
Separate training and held-out scenario seeds. Never interpret smoke rewards as a trained result.

## Task 1: Environment

Create `src/wild_drone/config.py`, `environment.py`, `__init__.py`, and `tests/test_environment.py`.
Interface: `WildfireFleetEnv(config: EnvConfig, render_mode=None)` implements reset, step,
joint encode/decode and `render()`; observation dictionary arrays are copied.
First write and run semantic tests (expected missing-module failures); implement transitions,
reward components and seeded scenario generation; run all environment tests and `check_env`.

## Task 2: Baselines

Create `src/wild_drone/baselines.py` and `tests/test_baselines.py`.
Interface: policies provide `act(env, observation) -> int`; random owns a separate RNG;
sweep owns waypoints; greedy uses observable hazard/age/battery; planner uses cloned states,
an independently reseeded generator, horizon/candidates/samples and a return-aware proposal.
First assert legal joint actions, baseline-safe returns and no changes to live environment
or live RNG during planning; implement and run environment plus baseline tests.

## Task 3: RL

Create `src/wild_drone/learning.py` and `tests/test_learning.py`.
Interfaces: full-state encoder, full-grid neural encoder, `train(algorithm, env_config,
train_config, output_dir)`, and `load_policy(path)` for deterministic inference.
First test discounted returns, known TD targets, finite updates, reproducible short training,
and checkpoint round trips. Implement DQN, REINFORCE and n-step A2C; run the complete suite.

## Task 4: Reproducibility and demo

Create CLI, configs, README, notebook walkthrough, CI workflow and `tests/test_cli.py`.
Commands: `demo`, `evaluate`, `train`, `compare`, `plot`. Save CSV plus JSON configuration,
dependency versions, training seed and revision; checkpoints include model/config metadata.
First test output files and disjoint train/evaluation seed validation. Implement commands,
run all tests, lint, Gymnasium checker and small training/evaluation runs for each algorithm.
Run the 40x40 baseline demo and inspect the GIF. Publish a branch and a reviewable PR; do not
merge it or present this starting release as a completed assignment experiment.
