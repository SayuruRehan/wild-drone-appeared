# Wild Drone Appeared: implementation design

The approved topic is coordinated, repeated wildfire inspection by three battery-constrained
drones, represented as a **fully observable finite-horizon MDP**. The controller sees the true
fire state; drone visits deliver fresh close-range inspections. Unknown-fire discovery is not
the objective. Success means a reproducible simulator and fair algorithm comparison, not an
assumed advantage for RL. This starting release supplies runnable methods and smoke checks;
long training, tuning and research conclusions follow after simulator review.

## Architecture and scope

Use a Python Gymnasium environment for one central controller. A PettingZoo multi-agent
interface would add complexity to a single joint policy without changing the approved task.
Keep simulation, baseline policies, neural RL, experiment runners and plotting separate.
No cloud services, real flight control, hidden maps or communication failures.

The grid is configurable: 20x20 for development, 40x40 for main experiments, 60x60 as a later
scalability test. All numerical settings are illustrative, dimensionless simulator values,
not calibrated metres, seconds or realistic wildfire physics. Fuel is integer-valued;
wind follows a discrete Markov process. Terrain affects fuel and spread susceptibility,
not flight accessibility. The central base has one charger and ground parking for all drones.

## Complete observation and actions

Observation is a dictionary: full-resolution burning, remaining fuel, terrain susceptibility,
static priority and inspection-age grids; every drone's row, column, battery and lost flag;
current wind, charger owner (including no owner), base position and remaining mission time.
Bounded normalisation retains every state variable; no cropped or downsampled policy input.
Config parameters are fixed and known within a run. All transition randomness uses the
environment's seeded generator. The generator's internal state is not an MDP state variable:
the transition kernel integrates over fresh random samples.

One discrete joint action encodes three commands in base five: stay, north, south, west,
east. There are 125 actions. Inspection is automatic after movement. Stay at base requests
charging; returning and recharging are not extra action categories. No safety shield or
action mask is included. Failed drones ignore subsequent commands.

## Exact transition order

1. Decode the joint action; propose simultaneous positions. Reject off-grid moves.
2. Block same-cell arrivals and direct swaps outside base. Repeat collision checks until
   stationary drones cannot be entered by other drones. Base co-location is permitted.
   Record blocked collision commands, not physical crashes.
3. Charge one eligible drone: retain a staying, not-full charger owner, otherwise select the
   lowest-battery eligible drone (identity breaks ties). Arrivals cannot charge until their
   next stay. Full drones release the charger. Waiting at base costs no flight energy.
4. Deduct movement/hover energy outside ground waiting, even for blocked move attempts.
   Zero battery away from base causes permanent loss; zero battery on arrival is safe.
5. Increment inspection ages, union the surviving drones' sensor footprints, and reset ages
   in the union. Duplicated simultaneous footprints earn no extra gain. No map-update reward
   is granted for failed drones. Record completed returns and successful charge completions.
6. Compute reward and timeliness metrics against the current hazard priorities. Reward is
   freshness improvement minus weighted overdue inspections, energy, collision attempts
   and new losses. Report each term; a penalty provides no safe-return guarantee.
7. Fire evolves synchronously: existing fires consume fuel and extinguish at zero; susceptible
   non-burning neighbours ignite with probabilities depending on current wind, remaining
   fuel and terrain. New fires start consuming fuel on the next transition. Change wind.
8. Advance mission time. All losses or the observed mission horizon terminate the episode.
   Fleet-wide loss also charges the remaining mission's full overdue penalty immediately,
   preventing early termination from avoiding future costs. Evaluation pads that remainder
   with zero freshness and full overdue status.
   The horizon is a genuine finite-mission terminal state, not an external truncation.

There are no collision counters or cumulative rewards that influence future transitions.
Charger ownership does influence future scheduling and is explicitly observed. Inspection
age starts at the inspection deadline and is capped/normalised at
`max(mission horizon, inspection deadline + 1)`. This preserves initially stale and later
overdue status even for smoke missions shorter than the deadline; saturation after that cap
is part of the simulator state definition and all freshness/gain terms already saturate.

## Controllers and scientific comparison

Baselines: seeded random, battery-aware partitioned sweep, priority/age greedy one-step
coordination, and a rolling-horizon stochastic shooting planner. The planner simulates its
candidate sequences on independent RNG streams: it knows the kernel but cannot inspect
the actual future RNG state. It has a bounded candidate budget, so it is not an exact optimum.
Report its horizon, candidates, rollout samples and decision time. All controllers receive
the same complete state and action/physical rules. Planner model access is an explicit
advantage over model-free RL and must be acknowledged in analysis.

DQN: joint-action Q network, replay, target network, epsilon exploration and TD updates.
REINFORCE: joint categorical policy, full-episode discounted returns and Monte Carlo gradient.
A2C: joint categorical actor, value critic and bootstrapped n-step updates. Policy reward
gradients use absolute-episode-time gamma^t weights. REINFORCE divides by the fixed mission
horizon, never the realised trajectory length; A2C divides by the configured rollout length.
Entropy is an optional local policy regulariser, separate from the environment return.
All are independently
implemented in PyTorch. They share the same full-grid convolutional feature architecture;
REINFORCE and actor-critic overlap in the broad policy-gradient taxonomy.

Use deterministic inference for final evaluation. Experiment rows identify training seed,
scenario seed, grid/config, algorithm and decision time. Hold out complete fuel/priority/
ignition scenarios; scenario IDs do not enter observations. Environment RNG use per step
does not depend on actions, supporting paired fire/wind scenarios for policies that survive.
Report across training seeds, not merely across episodes from one fitted model. Evaluate
reward, weighted freshness, overdue fraction, energy, losses, collisions, returns and charger
completions. The full 40x40 study is not complete until tuning, repeated training and held-out
tests are finished. Validate/recalibrate wildfire dynamics before practical or paper claims.

## Validation

Test boundaries, swaps, collision cascades, base co-location, charger exclusivity and owner
observability, zero-battery arrival, loss timing, duplicate scans, wind/fire reproducibility,
fuel exhaustion, state-space membership, episode endings and immutable observations.
Run Gymnasium's checker, short baseline episodes and learning updates for all three RL methods.
Render a labelled animation; inspect it visually. Keep generated pilot outputs under ignored
`runs/`; configuration and reproduction commands are versioned.

## Version-2 learning runtime

The simulator observation and physical/reward semantics remain unchanged. Neural input may
now derive three live-drone location maps and one base map from observed coordinates, alongside
the original five maps and 24-value vector. This is an ablatable feature transform using existing
full information. REINFORCE may fit a shared-encoder value baseline to full episode Monte Carlo
returns; its actor uses `return - detached value` and still has no TD bootstrap. A2C retains
n-step bootstrapping and DQN retains replay/target TD updates. Checkpoints explicitly identify
these architecture choices; version-1 policy files load with the old five-channel network.

Training state is saved at an episode boundary with optimizer, DQN target/replay, CPU/CUDA and
sampling RNGs, completed rows, validation selection and source/runtime history. The atomic
state is committed before regenerable exports so an interrupted first save remains recoverable. Cross-device or
cross-version bitwise equality is not promised. A no-op resume regenerates exports; a disconnect
may lose work since the last boundary. Checkpoint selection uses declared validation scenarios,
with final and selected models stored separately and final-test scenarios kept outside selection.
