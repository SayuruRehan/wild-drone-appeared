# Wild Drone Appeared: the team guide to the UI and simulation

Prepared 14 October 2026. Describes the mission-control implementation introduced in
commit `b0fbccb` on `sula`, with the frozen study and its separate follow-up experiments.
Read this alongside the [launch/API guide](mission-control.md),
[technical notebook](../Wildfire_Drone_Fleet.ipynb) and
[WhatsApp summary](mission-control-whatsapp.txt).

## Contents

1. [What the project is trying to do](#1-what-the-project-is-trying-to-do)
2. [The two ways to use the UI](#2-the-two-ways-to-use-the-ui)
3. [A first walkthrough for the team](#3-a-first-walkthrough-for-the-team)
4. [What each screen shows](#4-what-each-screen-shows)
5. [What happens in one simulator step](#5-what-happens-in-one-simulator-step)
6. [How batteries, charging and inspection work](#6-how-batteries-charging-and-inspection-work)
7. [What the controllers do](#7-what-the-controllers-do)
8. [How the animation works](#8-how-the-animation-works)
9. [How a new lab mission runs](#9-how-a-new-lab-mission-runs)
10. [How archived replays are built and checked](#10-how-archived-replays-are-built-and-checked)
11. [How to read the metrics and charts](#11-how-to-read-the-metrics-and-charts)
12. [The experiment story and its findings](#12-the-experiment-story-and-its-findings)
13. [Local setup, checkpoints and sharing](#13-local-setup-checkpoints-and-sharing)
14. [Common questions and troubleshooting](#14-common-questions-and-troubleshooting)
15. [Where to find the implementation](#15-where-to-find-the-implementation)

## 1. What the project is trying to do

One central controller coordinates **three drones inspecting a changing wildfire
landscape**. Each drone has a limited battery. All three share one charging station at
the central base. The challenge is to keep important locations recently inspected while
coordinating movement, returns and recharging.

The controller already sees the full simulator state: fire, fuel, terrain, priority,
inspection age, drone positions and batteries, wind, charger ownership and remaining
mission time. This makes it a **fully observable Markov decision process (MDP)**:
the observed state contains the information needed for the next transition, and the
mission has a fixed horizon.

A visit represents a fresh close-range inspection of a known location. The drones do
not extinguish fire or search for a hidden fire map. The simulation does not implement
fire damage to drones; permanent drone loss occurs when battery reaches zero away from
base. Terrain affects fire spread, rather than blocking flight.

The grid and energy values are illustrative simulator units. A cell is not a calibrated
physical distance, and a step is not a calibrated number of real-world seconds.

## 2. The two ways to use the UI

| Mode | What you can do | Where the states come from |
| --- | --- | --- |
| Hosted evidence showcase | Play archived missions, inspect drones and cells, compare controllers, read the research story and explore results | Verified exports of recorded study actions replayed through Python |
| Local simulation lab | Choose a controller and scenario, change allowed settings, generate a new mission, inspect and download it | A new run of the existing Python environment |

The hosted version is a static website. It includes the replay data and results needed
for the showcase. It does not run the Python simulator on the hosting service or contact
the visitor's laptop to start a local mission.

The local service serves the same frontend and also provides the simulation API. The
UI detects that API and enables the lab. A **new sandbox run** is an exploration: it
does not replace any published study result or launch RL training.

Private Site: https://wild-drone-mission-control.sulaweee.chatgpt.site/

The Site currently has owner-only access. ChatGPT sign-in is required, but a teammate
does not receive access simply by obtaining the link. Repository access and hosted Site
access are separate. Teammates with repository access can run the UI locally.

## 3. A first walkthrough for the team

Use this sequence for a short internal demonstration:

1. Open **Overview**. Explain the objective: sustained inspection, limited batteries,
   and a single charger. The preview is recorded evidence.
2. Open **Mission explorer**. Keep world **22000**, with **Greedy** and **DQN seed 0**.
   The two maps use the same scenario and shared playback timeline.
3. Press Play. Watch the three drone colours, battery bars and routes. Pause at an
   interesting step or drag the timeline. The underlying mission is unchanged.
4. Switch to **Freshness**. Explain that recently inspected cells are brighter, and
   inspections become stale again. Visible movement alone is not the objective.
5. Select **Inspect DQN · seed 0**. Open its allowed-action decision details after
   step 0. These show recorded action scores and the feasible-action set.
6. Select a drone card to highlight its recent trail. Inspect a map cell by clicking
   it, or enter its row and column to inspect fuel, priority and inspection age.
7. Open **The research**, then **Results & evidence**. Explain the difference between
   a single displayed mission and the full 260-mission aggregate comparison.
8. If running locally, use **Simulation lab** to run a new baseline mission. Show its
   sandbox label and explain that it is separate from the frozen evidence.

Do not use a favourable individual mission as proof that an algorithm succeeds across
worlds. The results screen provides the repeated-seed evidence needed for that discussion.

## 4. What each screen shows

### Overview

This introduces the task, provides an interactive recorded mission, and highlights the
main finding: the feasibility rules helped fleet survival, while useful learned inspection
remained weak. The Greedy/DQN percentages shown here come from the full frozen study.

### Mission explorer

**Controller** selects a baseline or a fitted learner. Learners are labelled with their
training seed, identifying a particular trained model. **Scenario** selects a world seed,
identifying the environment realization. These are different kinds of seed.

The showcase contains three worlds—22000, 22009 and 22019—for all nine fitted learners and
four baselines, giving **39 replays**. It defaults to Greedy versus DQN seed 0 on world 22000.
Planner is available in the local lab and follow-up results; it is not one of these 39
frozen-main-study replay options.

The controls provide Play/Pause, Restart, one-step advance, timeline seeking, and playback
speed. In the explorer, keyboard Space toggles playback and Left/Right arrows step when
focus is outside a form control or button. Mobile comparison maps stack vertically.

| Map element or layer | Meaning |
| --- | --- |
| Fire & fuel | Current burning cells, remaining fuel and terrain appearance |
| Orange cells | Active fire in the displayed simulator state |
| Freshness | Inspection recency: bright is recently inspected; dark is stale |
| Priority zones | Static higher-priority areas, highlighted with a warm tint |
| Shared base | The central location where drones can park and charge |
| Sensor footprint | The square region inspected by each surviving drone |
| Flight trails | Up to the most recent 70 transitions of a drone's route |
| Wind | Current discrete wind direction, or calm |

At base, drone icons have small display offsets so all three can be seen. Their actual
grid coordinates remain the same base cell, as shown in telemetry. The offsets do not
represent different simulator positions.

The inspector can follow either comparison controller. Clicking its map or using its
**Inspect** button chooses the corresponding telemetry. Drone cards show battery,
position, command, distance to base and return margin. A selected card highlights that
drone's trail and dims the others. Cell coordinates are zero-based: a 40×40 grid uses
rows and columns 0–39.

**What happened this step** describes observable events: returns, charging, recharge
completion, blocked movement and losses. The reward breakdown lists the actual terms
for the selected transition.

Archived RL transitions also include the 125 recorded scores and their allowed actions.
The UI displays the top allowed scores and identifies the selected action. These are
DQN Q estimates or actor logits, depending on the algorithm. They show the deployed
selection rule; they are not explanations of the network's internal reasoning or
calibrated probabilities of success. Step 0 has no preceding decision.

### The research

This explains the progression from the MDP and baselines to the three learners, failure
diagnostics, feasibility masking and the frozen comparison. Expandable details link to
the implementation and protocols. It is a guided account of the existing work, including
negative findings.

### Results & evidence

This presents full-study means and sample standard deviations, the nine fits' raw training
curves, separate follow-up tables, source identity and evidence downloads. It supports
metric and algorithm selection, and switching among planner, sensitivity and learning-rate
follow-ups. Chart hover exposes values where supported.

### Simulation lab

The lab exposes controller, world seed, 20×20/40×40 preset, battery capacity, recharge
rate and spread probability. It supports baseline experiments and verified frozen RL
checkpoints. The current sandbox ranges are battery **40–240**, recharge **1–16**, and
spread probability **0–0.12**; these are UI bounds, not experimentally validated real-world
ranges. New runs are labelled separately and completed runs can be downloaded as JSON.

## 5. What happens in one simulator step

All three drones receive their commands together. A joint action is one number encoding
three choices from **stay, north, south, west and east**: `5 × 5 × 5 = 125` possibilities.
Inspection is automatic, and staying at base requests charging.

| Order | What Python does | Why it matters |
| --- | --- | --- |
| 1 | Decode the joint action and propose simultaneous positions | The whole fleet is controlled in one decision |
| 2 | Reject off-grid moves and resolve same-cell arrivals, swaps and collision cascades outside base | A blocked command does not teleport or crash a drone; base co-location is allowed |
| 3 | Allocate the single charger to an eligible staying drone | Arriving this step is insufficient to start charging immediately |
| 4 | Deduct movement/hover energy and mark new battery losses | Commands can consume energy even if the attempted move was blocked |
| 5 | Increase inspection ages, then reset cells in surviving sensor footprints | Overlapping footprints are combined; duplicated coverage earns no duplicate gain |
| 6 | Compute inspection metrics and the reward components | Metrics use the hazard priorities at this point in the transition |
| 7 | Consume fuel, evolve fire synchronously and update wind | The fire follows its stochastic kernel independently of the drone actions |
| 8 | Advance elapsed time and decide whether the mission ends | The horizon or loss of the entire fleet terminates the mission |
| 9 | The UI adapter copies the resulting state into a snapshot | The browser receives data representing what actually happened |

There is a subtle timing distinction: the displayed snapshot's fire map is **after fire
evolution**, while the transition's reported freshness and reward use the weights **before
that evolution**. The UI preserves the simulator's reported metrics rather than silently
recalculating them from the later map.

If fire burns out before the horizon, the mission continues: priority-area inspection
still matters. If one drone is lost, surviving drones continue. If all three are lost,
the mission ends and its unserved remainder is included in the accounting.

## 6. How batteries, charging and inspection work

### Main and development presets

| Setting | Main 40×40 preset | Development 20×20 preset |
| --- | ---: | ---: |
| Horizon | 300 steps | 160 steps |
| Battery capacity | 160 units | 80 units |
| Recharge rate | 8 units/step | 5 units/step |
| Sensor radius | 2 cells | 1 cell |
| Inspection deadline | 30 steps | 25 steps |
| Maximum initial fuel | 180 | 100 |
| Initial fires | 3 | 3 |
| Base spread probability | 0.04 | 0.04 |

For both presets, a non-stay command costs 1 energy unit and hovering away from ground
waiting costs 0.25. A blocked move attempt still incurs its command's energy cost.
Staying on the ground at base consumes no flight energy. A drone with zero battery on
safe arrival at base survives; zero battery away from base causes permanent loss.

One charger can charge one eligible drone per step. An eligible current owner retains
it while staying and not full. Otherwise, the lowest-battery eligible drone is chosen,
with drone identity breaking ties. A full recharge releases the charger. Other drones
can wait at base with no ground-stay energy cost.

Telemetry's **return margin** is:

```text
Manhattan distance = |drone row − base row| + |drone column − base column|
Return margin = battery − distance × move cost
```

For example, 15 battery units at a distance of 13 cells gives a margin of 2. That is
little spare energy beyond the direct route. Collisions, waiting or other commands can
spend additional energy; the margin is a diagnostic, not a proof of safe return.

The 40×40 sensor radius of 2 normally covers a 5×5 square, clipped at grid edges. The
20×20 radius of 1 normally covers a 3×3 square. Each inspected cell's age resets to zero,
then increases again on later steps unless revisited. Surviving drones at base also
inspect their footprint.

Fire evolves across four-neighbour cells using fuel, terrain, wind and seeded random
draws. Existing burning cells consume fuel; fuel-depleted fires extinguish. New ignitions
start consuming fuel on the following transition. The fleet's inspections do not alter
that fire evolution.

## 7. What the controllers do

| Controller | How it chooses commands |
| --- | --- |
| Random | Uniformly chooses from all 125 joint commands, without a battery-return rule |
| Masked random | Uniformly chooses among actions allowed by the feasibility mask |
| Sweep | Follows partitioned serpentine patrol targets with battery-aware proposals |
| Greedy | Scores immediate priority/inspection-age gain over its feasible proposals; a small tie preference reduces movement |
| Planner | Uses bounded, independently sampled hypothetical futures and replans every step |
| DQN | Uses a learned Q network; frozen evaluation selects the highest-scoring allowed joint action |
| REINFORCE | Learns a categorical actor from complete discounted episode returns, with a learned Monte Carlo value baseline |
| A2C | Learns an actor and value critic using synchronous n-step bootstrapped updates |

Each learner controls the entire three-drone fleet. There are not three independent
drone agents learning separately. The shared neural representation combines full-grid
maps with drone/global values; its default spatial encoding adds observed drone/base
location channels. Scenario IDs are not policy inputs.

### What the feasibility mask contributes

The mask removes joint actions that fail its battery, boundary or collision rules.
Low-battery return conflicts can make the baseline proposal function fall back to all-stay;
the masked controllers add an emergency-return rule that prioritizes critical drones.

Greedy/Sweep/Planner use their native proposal logic; they do not automatically receive
every detail of the masked learner's emergency-return handling. This distinction matters
when interpreting the follow-up return-deadlock failures.

The mask is a deterministic control rule, not a learned achievement. Masked random is
therefore an essential comparison: it receives the same rule without a trained network.
The experiment can distinguish the benefit of feasible actions from the benefit of learning.

Training, inference and playback are separate activities. Training updates network
weights. A new local RL mission performs inference with a verified fixed checkpoint.
Archived playback renders states reconstructed from saved actions. Pressing Run mission
or Play does not train a new network.

## 8. How the animation works

The animation consists of **simulator-state snapshots rendered by Canvas**. The stored
frames contain structured data, not a sequence of screenshots.

```text
Python environment + controller
           │
           ▼
  Actual transition and state snapshot
           │
           ▼
  Recorded replay file OR local API response
           │
           ▼
  Browser snapshot buffer + playback index
           │
           ▼
  Canvas map, telemetry, events and charts
```

Each snapshot includes elapsed step, burning/fuel/inspection-age arrays, drone positions,
batteries, lost flags, wind, charger owner, commands, reward terms, inspection metrics and
cumulative counters. Archived RL snapshots also include decision scores and allowed actions.
Static terrain, priorities, base coordinates and configuration are in the mission packet.

Step 0 is the initial state. A complete 300-step mission therefore has **301 snapshots**;
a complete 160-step mission has **161**. An early fleet loss produces fewer actual snapshots.

Canvas paints map cells, the base, sensor footprints, trails and drone icons using the
chosen snapshot. React updates the selected step and the corresponding text/card values.
The browser interpolates drone icon positions over a nominal **160 ms** visual transition,
using animation callbacks. Reduced-motion preference makes this transition immediate.
Interpolation also applies when the selected state changes through seeking; intermediate
icon positions are presentation effects, not extra simulator states or decisions.

There are three separate clocks:

| Clock | What it controls |
| --- | --- |
| Simulator steps | The discrete transitions that change batteries, maps, reward and mission time |
| Python computation time | How quickly the CPU computes those transitions; the planner can take longer |
| Browser playback/rendering time | How quickly stored snapshots are selected and visually drawn |

The playback timer targets `250 / speed` milliseconds per simulation step:

| UI speed | Target snapshot advances per second |
| --- | ---: |
| 1× | 4 |
| 2×, the default | 8 |
| 4× | 16 |
| 8× | 32 |

These are display rates, not physical flight speeds or guaranteed browser frame rates.
A 300-step replay takes roughly 37.5 seconds at the default rate if the browser keeps up.
Python may finish computing a lab mission sooner than that. Increasing playback speed
does not increase battery drain per simulator step or change the actions, fire or results.

Pause holds a selected snapshot; seeking selects another stored state. These do not rewind
the Python worker or undo its already computed transitions. In a running lab mission,
generation can continue while playback is paused. Leaving the lab requests cancellation
of its job; the current simulator step/decision finishes before cancellation takes effect.

For archived early fleet loss, the explorer holds the final actual map and marks the
remaining timeline unserved. Its padded freshness is zero. This does not generate fictional
post-loss actions or imply that the failed fleet keeps flying.

## 9. How a new lab mission runs

1. The local UI reads `/api/capabilities` to discover available controllers.
2. Run mission sends the controller, world seed, grid preset, battery, recharge rate and
   spread probability to the Python API.
3. The API validates the request and returns a unique job ID. Invalid settings or
   unavailable controllers do not start a job.
4. A background worker creates its own environment and controller. Reset initializes
   the world using the selected seed, and the initial snapshot is stored.
5. The worker repeatedly obtains `policy.act(env, observation)`, calls `env.step(action)`,
   and stores the actual resulting state. It does not pause to match animation speed.
6. The browser polls the job, requesting only frames newer than its last received step.
   While the job is queued/running, it schedules another poll after about 300 ms; request
   latency also contributes to the actual polling interval.
7. New frames are appended to the browser buffer. Playback can wait for more frames if
   it reaches the available buffer while generation is still running.
8. Completion supplies full-horizon mission metrics. Cancellation produces a cancelled
   run without a completed summary. A failure supplies an error instead of an invented result.

The job statuses are queued, running, completed, cancelled and failed. There are two
simulation workers and at most eight retained jobs. Completed jobs expire after 30 minutes,
with cleanup on a subsequent job creation; capacity pressure can evict an older finished
job. Jobs are held in memory and disappear on server restart. Download a completed run
if you want to keep it.

### A concrete state you can inspect yourself

Select **Greedy**, world **22000**, and seek to **step 145**. The exported evidence gives:

| Drone | Actual position (row, column) | Battery | Distance to base | Return margin |
| --- | --- | ---: | ---: | ---: |
| 01 | 33, 20 | 15 | 13 | 2 |
| 02 | 13, 18 | 15 | 9 | 6 |
| 03 | 21, 8 | 15 | 13 | 2 |

The base is at 20, 20. All three drones have the same remaining battery, but Drone 02
has more direct-route energy spare because it is closer to base. The UI marks Drone 01
and Drone 03 with critical return margins. At this step, reported freshness is about
21.47% and cumulative return is −8.68. Those are this mission's selected-step values;
Greedy's 22.08% result elsewhere in the guide is the full-study mean.

The current live-lab snapshots provide state/events/rewards. They do not export the full
125-score decision vectors that are available in the archived RL decision inspector.

## 10. How archived replays are built and checked

The exporter reads the preserved main-study action traces and source evidence. It resets
the Python environment with the frozen configuration and world seed, then executes the
recorded action sequence. This reconstructs the actual display states without needing
the trained policy weights.

For every selected transition, it checks the pre/post observation hashes, action feasibility,
reward, every reward component, inspection metrics and termination. For RL, it checks the
saved feasible mask and that the recorded action is the first-index masked argmax of the
saved scores. The final full-horizon summary must match the archived mission row.

The **39 UI replays cover 11,389 actual transitions**. Each packet is JSON compressed
with gzip. The browser downloads the selected file, checks its SHA-256 against the evidence
catalogue, decompresses it and parses its data. It keeps a small in-memory cache of up to
six decoded missions, so the website does not eagerly load every mission.

There are two verification scopes to distinguish:

- **UI export verification:** reconstructs selected states and checks saved evidence. It
  does not rerun neural inference to prove how the weights generated those saved scores.
- **Original independent main-study audit:** covers all 260 missions and 75,911 actions,
  including checkpoint identity and neural score reconstruction. That larger audit is
  preserved in the archive; the UI reports its recorded result.

Full-study means and sample SDs are independently recomputed from the **260 mission rows**
during export and checked against the archived summary. Follow-up table inputs are copied
from their separate committed CSVs, with their source hashes recorded in the evidence index.

## 11. How to read the metrics and charts

### Freshness and overdue inspections

For a cell with inspection age `a` and deadline `d`:

```text
Cell freshness = 1 − min(a / d, 1)
```

With deadline 30, age 0 is fully fresh, age 15 has freshness 0.5, and age 30 or more has
freshness 0. A cell becomes overdue when its age is **strictly greater** than the deadline.
At mission start, cells have age equal to the deadline: due for inspection, with zero
freshness, but not yet overdue.

Location weights combine static priority with current burning/frontier status. Burning
cells receive three times their static weight; neighbouring susceptible frontier cells
receive twice their static weight. Weighted freshness and overdue fraction divide their
weighted totals by the sum of weights. Therefore freshness is **inspection recency**, not
the percentage of distinct land area covered or fire extinguished.

The mission average uses the whole configured horizon. If a fleet fails at step 200 of
300, the remaining 100 steps contribute zero freshness and full overdue status. This
prevents early failure from looking good by ending the measurement early.

### Reward, return and counters

Reward combines inspection-freshness gain with overdue, energy, collision-attempt and
new-loss penalties. Total fleet failure additionally charges the remaining overdue mission
cost. **Return** is the sum of these rewards over the mission. The inspected-cell gain term
is not identical to the displayed whole-map weighted-freshness metric.

Negative returns are expected when accumulated costs exceed positive gains. A return of
−16 is higher/better than −25. The UI's return bars extend left from zero, and their actual
signed values are printed beside them.

| UI value | Correct interpretation |
| --- | --- |
| Current freshness | Quality at the selected recorded transition, with failed remainder shown unserved |
| Mission freshness | Mean quality across the full horizon, including an early-failure remainder |
| Mission return so far | Cumulative reward up to the selected state |
| Energy used | Cumulative energy spent, rather than the batteries' current charge |
| Safe returns | Count of returns to base; one drone can return multiple times |
| Charge completions | Count of completed recharges |
| Collision attempts | Blocked collision commands, rather than physical crashes |
| Drone losses | Permanently lost drones, from 0 to 3 |
| Return margin | Direct-route energy surplus, rather than a safety guarantee |

Zero losses does not establish that all drones finished at base or inspected effectively.

### Uncertainty and training curves

For each learner, first average its 20 worlds within each fitted training seed, then
calculate the mean and sample SD across the **three fitted-seed means**. For a baseline,
calculate mean and sample SD across **20 worlds**. Those error bars represent different
variation sources. They are not confidence intervals or statistical significance tests.

Training curves show per-episode training return for each fitted seed. They are not held-out
test curves. Hover uses chart indices: index 0 corresponds to training episode 1. A training
trend alone does not prove generalization, useful coordination or convergence.

## 12. The experiment story and its findings

The project began with the simulator and conventional baselines, then independently
implemented DQN, REINFORCE and A2C. Early unmasked pilots showed poor return/recharge
behaviour. Stability, exploration and supervised capability diagnostics investigated
possible causes. Historical revisions and their negative results remain preserved.

Some 20×20 supervised inspection studies improved development-world performance with
broader training coverage. Their results concern supervised scorers and particular
development settings; they are not successful RL or a controlled grid-scaling result.
The 40×40 supervised pilot passed ranking checks but failed the full-mission criterion;
the loss-only follow-up also failed its declared gates.

The frozen **11 October 2026 main comparison** used DQN, REINFORCE and A2C, each with
three independent training seeds and 300 episodes. All nine fits completed 90,000
interactions apiece. Each final checkpoint ran 20 fresh worlds, alongside four baselines:
180 learner missions plus 80 baseline missions, totaling 260.

No best-checkpoint validation selection, supervised warm start, reward shaping or
post-result extra fitting was used for this main comparison.

| Controller | Mean freshness | Mean return | Mean losses |
| --- | ---: | ---: | ---: |
| DQN | 8.44% | −24.79 | 0 |
| REINFORCE | 6.90% | −25.81 | 0 |
| A2C | 8.43% | −24.83 | 0 |
| Greedy | 22.08% | −16.10 | 0 |
| Sweep | 6.49% | −26.07 | 0 |
| Random | 5.96% | −42.87 | 3 |
| Masked random | 8.38% | −25.46 | 0 |

The masked learners preserved the fleet in this sample, but masked random did too. DQN
and A2C were close to masked random in freshness, and REINFORCE was lower. Greedy inspected
substantially more effectively. None of the nine fitted policies passed either complete
predeclared performance gate.

The learned-contribution gate requires every fit to exceed masked random by at least
0.01 freshness and 0.5 return, with no worse losses/fleet failures. A separate Greedy gate
requires at least 90% of Greedy freshness and comparable overdue/safety metrics. These are
project capability criteria, not assignment grading thresholds.

### Follow-up experiments shown separately

- **Planner:** matched Greedy's default-setting freshness, with a slightly worse mean
  return and higher serial decision cost. This is a model-based controller comparison.
- **Sensitivity:** evaluated fixed policies under changed battery, recharge and fire-spread
  settings without retraining. Four Greedy fleet failures outside the default revealed a
  return deadlock. Some apparent learner improvements may reflect changes to the mask;
  that proposed mechanism was not separately established.
- **Learning rates:** three rates per algorithm, one training seed and 20 validation worlds.
  No tested rate closed the Greedy gap. One seed cannot establish reliable rate ordering
  or rule out learning rate as a contributing cause. These same-machine runs do not
  reproduce the archived Linux training trajectories.

The accurate team message is: **the simulator and comparison are complete; mask-assisted
survival improved, while a reliable learned inspection advantage was not demonstrated.**
The fixed budget does not prove convergence, and the historical 20×20/40×40 comparison
does not isolate a grid-size effect. No 60×60 results are claimed.

## 13. Local setup, checkpoints and sharing

From the cloned repository root, use Python 3.11+ and Node 22.12+. On macOS/Linux,
activate the project's virtual environment first; create it if it is missing.

The UI is currently on **`sula`**, with [PR #38](https://github.com/SayuruRehan/wild-drone-appeared/pull/38)
open against `master`. A fresh clone for this UI should select that branch:

```bash
git clone --branch sula https://github.com/SayuruRehan/wild-drone-appeared.git
cd wild-drone-appeared
```

If you already have a checkout, preserve your local changes before switching branches.
The commands below assume you are in a checkout containing the UI.

```bash
# If you do not already have a virtual environment:
python3 -m venv .venv

source .venv/bin/activate
python -m pip install -e '.[ui,rl]'
cd web
npm ci
npm run build
cd ..
wild-drone-ui
```

Open http://127.0.0.1:8000. If the environment and frontend are already installed/built,
running `.venv/bin/wild-drone-ui` from the repository root is sufficient on macOS/Linux.
Windows users activate with `.venv\Scripts\Activate.ps1`, then use the same Python/npm
commands and `wild-drone-ui` launch command.

The five baseline controllers require no learned weights. Live frozen RL requires
original checkpoint files downloaded using the [archive instructions](experiments/rl-main-2026-10-11/README.md).
Point the service at the fit directory:

```bash
wild-drone-ui --checkpoint-root runs/rl-main-2026-10-11/fits
```

The expected layout is `fits/dqn/seed-0/policy.pt`, repeated for all methods and seeds.
The service checks the checkpoint's SHA-256 and algorithm, seed, environment and training
configuration. It verifies the bytes again when starting a new RL run. Different or
missing files remain unavailable. Frozen live learners require the unchanged 40×40 settings;
baseline sandbox controls support parameter changes. The UI does not automatically
download checkpoints or upload custom weights.

Archive playback works without these checkpoint files because its recorded actions have
already been reconstructed and exported. Code and archived evidence are shared through
the repository; hosted Site permissions are a separate access setting. This guide and
the WhatsApp text can be shared with the team without implying that the Site is public.

## 14. Common questions and troubleshooting

| Question or symptom | Explanation or next step |
| --- | --- |
| Is this a video? | It is structured state data drawn by Canvas. Every stored state comes from Python |
| Is the Python simulation running at 60 steps per second? | Simulation, snapshot playback and screen rendering have separate rates |
| Why does a lab run finish before playback ends? | Python computes as quickly as its controller/environment allow; playback is paced separately |
| Does 8× playback drain batteries faster? | It displays more recorded steps per second; each step's energy cost and mission outcome are unchanged |
| Can a drone sit over fire? | The model allows it. Fire does not directly damage drones in this implementation |
| Why is a drone waiting at base? | It may be staying to request charging or waiting for the shared charger |
| Why did a move command leave the position unchanged? | A boundary or collision rule can block movement; command energy may still be charged |
| Why does safe random perform close to an RL learner? | The mask itself contributes safety and feasible movement; learning must add an advantage beyond it |
| Why are RL options disabled in the local lab? | Required frozen checkpoints are missing or failed identity checks |
| Why are parameter controls locked for a frozen learner? | The live loader requires the environment used to train that checkpoint |
| The hosted lab says a local simulator is required | Start the Python service and open the local address rather than expecting hosted Python execution |
| A teammate cannot open the Site | It is owner-private; their ChatGPT sign-in alone does not authorize viewing |
| The local page is unavailable | The service may be stopped. Run `wild-drone-ui`; use `--port 8001` if port 8000 is occupied |
| Replay integrity check failed | Reload the matching current build/data. Do not bypass the checksum check |
| A mission disappeared | In-memory jobs can expire, be evicted or disappear on restart. Download completed runs |
| A cancelled planner takes a moment to stop | Cancellation is cooperative and waits for the current decision/step to finish |

## 15. Where to find the implementation

| Area | Source |
| --- | --- |
| Environment, complete state and transition order | [environment.py](../src/wild_drone/environment.py), [design.md](design.md) |
| Default settings | [config.py](../src/wild_drone/config.py) |
| Conventional controllers | [baselines.py](../src/wild_drone/baselines.py) |
| Neural architecture and learning updates | [learning.py](../src/wild_drone/learning.py) |
| Feasibility and emergency-return masks | [masked_learning.py](../src/wild_drone/masked_learning.py) |
| Snapshot/event serialization and full-horizon summary | [state.py](../src/wild_drone/webapp/state.py) |
| Local API, job management and checkpoint verification | [server.py](../src/wild_drone/webapp/server.py) |
| Replay export and aggregate recomputation | [export_web_evidence.py](../scripts/export_web_evidence.py) |
| Page behaviour, polling and playback | [App.tsx](../web/src/App.tsx) |
| Canvas drawing and interpolation | [Map.tsx](../web/src/Map.tsx) |
| Browser gzip/checksum/cache and state selection | [data.ts](../web/src/data.ts) |
| Learning/mission line charts | [Charts.tsx](../web/src/Charts.tsx) |
| Machine-readable UI evidence | [evidence.json](../web/public/data/evidence.json) |
| Full frozen comparison and independent audit | [main-study archive](experiments/rl-main-2026-10-11/README.md) |
| Follow-up experiments and limitations | [reference experiments](experiments/rl-followups-2026-10-11/README.md) |

The UI presents the simulator and evidence in an explorable form. It does not change the
underlying mission objective, physical rules, learning algorithms or historical outcomes.
