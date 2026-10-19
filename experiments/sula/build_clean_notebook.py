# ruff: noqa: E501  (markdown table rows cannot be wrapped)
"""Build Wildfire_Drone_Fleet_clean.ipynb: one self-contained, rubric-ordered notebook.

Core definitions are copied verbatim from src/wild_drone (via the parser, so no hand
transcription); explanatory and analysis cells are reused from Wildfire_Drone_Fleet.ipynb.
Run from the repository root:  python experiments/sula/build_clean_notebook.py
Then execute:  jupyter nbconvert --to notebook --execute --inplace Wildfire_Drone_Fleet_clean.ipynb
"""

import ast
import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[2]
SOURCE = nbformat.read(ROOT/'Wildfire_Drone_Fleet.ipynb', as_version=4)
OUT = ROOT/'Wildfire_Drone_Fleet_clean.ipynb'
AUTHOR = '**Author: _(name)_**'


def original(index):
    return SOURCE.cells[index].source


def edited(index, *replacements):
    text = original(index)
    for old, new in replacements:
        if old not in text:
            raise AssertionError(f'cell {index}: expected text not found: {old[:60]!r}')
        text = text.replace(old, new)
    return text


def extract(path, names, drop_methods=()):
    """Verbatim top-level definitions from a source file, in the requested order."""
    text = (ROOT/path).read_text()
    lines = text.splitlines()
    found = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        else:
            continue
        if name not in names:
            continue
        start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
        removed = set()
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in drop_methods:
                    removed.update(range(item.lineno-1, item.end_lineno+1))
        body = [line for number, line in enumerate(lines[start-1:node.end_lineno], start)
                if number not in removed]
        while body and not body[-1].strip():
            body.pop()
        found[name] = '\n'.join(body)
    missing = set(names) - set(found)
    if missing:
        raise AssertionError(f'{path}: definitions not found: {sorted(missing)}')
    return '\n\n\n'.join(found[name] for name in names)


def heading(number, title, intro=None):
    text = f'## {number}. {title}\n\n{AUTHOR}'
    return new_markdown_cell(text + (f'\n\n{intro}' if intro else ''))


def strip_heading(text):
    """Drop the original numbered H2 heading line; the new notebook renumbers sections."""
    return re.sub(r'\A## [^\n]*\n+', '', text)


cells = []
md = lambda text: cells.append(new_markdown_cell(text))  # noqa: E731
code = lambda text: cells.append(new_code_cell(text))  # noqa: E731

# ---------------------------------------------------------------------------------- 0
md("""# Coordinated Wildfire Inspection with a Battery-Constrained Drone Fleet

**COMP6008 Reinforcement Learning — group project notebook**

| Section | Contributor(s) |
| --- | --- |
| 1–4 Setup, problem, MDP, environment | _(name)_ |
| 5–7 Baselines, feasibility mask, RL agents | _(name)_ |
| 8–13 Experiments, tuning, results, review | _(name)_ |

One central controller commands three drones that repeatedly inspect a changing fire field
and share one charger. We model the task as a fully observable finite-horizon MDP, implement
the environment as a Gymnasium environment, and compare three RL agents — **DQN**,
**REINFORCE with a learned baseline** and **synchronous A2C** — against conventional
controllers.

**Main finding.** With a feasibility mask (battery and collision rules applied to the action
choice), all three learners kept every drone alive in every final 40×40 mission, where the
unmasked learners and unrestricted random lost the fleet. The learned inspection quality,
however, stayed at the level of random feasible actions and far below the one-step Greedy
controller. Sections 8–12 document how we reached this result and why.

All code that runs in this notebook is defined in it. Training is opt-in (Section 1 flags);
by default the notebook reloads the archived results of the completed experiments.""")

# ---------------------------------------------------------------------------------- 1
cells.append(heading(1, 'Setup and reproduction'))
md("""Requirements: Python 3.11 or 3.12. From the repository root:

```bash
python -m pip install -r requirements.txt
python -m jupyter notebook Wildfire_Drone_Fleet_clean.ipynb
```

In Google Colab, clone the repository into `/content/wild-drone-appeared`, run
`%pip install -r requirements.txt`, and restart the kernel once.

**Data.** Archived results are read from the repository (no download needed):
`configs/` (environment and training settings), `docs/experiments/final-2026-10-05/`
(20×20 development study), `docs/experiments/rl-main-2026-10-11/` (frozen 40×40 study) and
`experiments/lithma/results/` (40×40 learning-rate sweep, sensitivity and planner runs).

**Flags.** With the defaults below, *Run All* takes a few minutes on a CPU and trains only
tiny demonstration models. Set `RUN_TRAINING = True` to retrain one frozen 40×40 fit
(about an hour on one CPU core) or `RUN_SWEEP = True` to rerun the 40×40 learning-rate sweep
(nine fits).""")
code("""import copy, json, random, sys, tempfile, time
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from IPython.display import display
from torch import nn
from torch.distributions import Categorical

RUN_TRAINING = False   # retrain one frozen 40×40 fit (slow)
RUN_SWEEP = False      # rerun the 40×40 learning-rate sweep (very slow)

candidates = [Path.cwd(), Path.cwd().parent, Path('/content/wild-drone-appeared')]
REPO = next((p for p in candidates if (p/'configs/rl-main-frozen.json').exists()), None)
if REPO is None:
    raise FileNotFoundError('Open this notebook from the repository root (see setup above).')
HISTORICAL = REPO/'docs/experiments/final-2026-10-05'   # 20×20 development study
MAIN = REPO/'docs/experiments/rl-main-2026-10-11'       # frozen 40×40 main study
FOLLOW_UP = REPO/'experiments/lithma/results'           # 40×40 sweep, sensitivity, planner
ALGORITHMS = ['dqn', 'reinforce', 'a2c']
POLICIES = ALGORITHMS + ['greedy', 'sweep', 'random', 'masked_random']
FROZEN_SOURCE = '855e5510ff46b16082ab7d551e7c228acab6f572'  # code revision of the main study

pd.set_option('display.max_columns', 20)
pd.set_option('display.precision', 5)
plt.rcParams.update({'figure.dpi': 90, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 9})


def finish_plot(fig):
    fig.tight_layout()
    plt.show()
    plt.close(fig)


print('Python', sys.version.split()[0], '| torch', torch.__version__, '| numpy', np.__version__)""")

# ---------------------------------------------------------------------------------- 2
problem, mdp = original(4).split('\nLet $S_t', 1)
cells.append(heading(2, 'Problem and why reinforcement learning'))
md(strip_heading(problem).strip() + """

**Why it is hard.** The fleet decision is joint: 5 commands per drone give 125 joint actions
per step. Each drone must return before its battery empties, but the cost of flying too far
appears many steps later as a permanent loss. One charger creates contention, the fire
spreads stochastically, and missions last 160–300 steps.

**Why RL.** The task is a sequential decision problem under uncertainty with a clear
numeric objective, so an agent can in principle learn a policy that balances inspection,
travel and recharging from interaction alone. Because the full state is known, we also
compare against non-learning controllers that use the same information.""")

# ---------------------------------------------------------------------------------- 3
cells.append(heading(3, 'Markov decision process'))
md('### State, observation and action\n\nLet $S_t' + mdp)
md(original(5).replace('### Transition order and reward', '### Transition model and reward'))

# ---------------------------------------------------------------------------------- 4
cells.append(heading(4, 'Environment implementation', """The environment follows the
`gymnasium.Env` interface. The cells below are the implementation used for every experiment;
comments refer to the transition steps 1–5 of Section 3."""))
code('# Simulator settings (dimensionless, illustrative values)\n' +
     extract('src/wild_drone/config.py', ['EnvConfig']))
code(extract('src/wild_drone/environment.py',
             ['MOVES', 'WIND', 'decode_action', 'encode_action', 'neighbour_count',
              'WildfireFleetEnv'], drop_methods=('render',)))
code("""from gymnasium.utils.env_checker import check_env

frozen = json.loads((REPO/'configs/rl-main-frozen.json').read_text())
development = json.loads((REPO/'configs/development.json').read_text())
main_env = EnvConfig(**frozen['environment'])
check_env(WildfireFleetEnv(main_env), skip_render_check=True)
assert all(encode_action(decode_action(a)) == a for a in range(125))
print('Gymnasium environment check and all 125 action encodings passed.')""")
code(original(7))

# ---------------------------------------------------------------------------------- 5
cells.append(heading(5, 'Baselines and the feasibility mask'))
md(edited(8, ('## 3. Baselines and feasibility mask\n\n', ''),
          ('The main 40×40 comparison uses the first four. All stay and the rolling-horizon '
           'planner remain in the historical evidence; they were not silently added to the '
           'frozen final comparison.',
           'The frozen 40×40 comparison uses the first four; the rolling-horizon planner was '
           'evaluated on the same worlds in a separate follow-up (Section 10).')))
md("""**Why a mask?** The learners originally had no battery rule. In the 20×20 development
study (Section 8) the unmasked DQN, REINFORCE and A2C lost 1.7–3 drones per mission, while
every conventional controller kept its fleet because `candidate_actions` filters out moves
that would strand a drone. The mask gives the learners the same feasibility rule, so the
comparison measures inspection decisions rather than who was told about batteries. The idea
follows *shielding* in safe RL (Alshiekh et al., 2018).""")
code(extract('src/wild_drone/baselines.py',
             ['COMMANDS', 'candidate_actions', 'RandomPolicy', 'GreedyPolicy', 'SweepPolicy',
              'RollingHorizonPolicy']))
code(extract('src/wild_drone/masked_learning.py',
             ['action_mask', 'emergency_return_mask', 'apply_mask', 'masked_max',
              'MaskedRandomPolicy']))
code("""env = WildfireFleetEnv(main_env)
env.reset(seed=0)
mask = action_mask(env)
print('Feasible joint actions at reset:', int(mask.sum()), 'of 125')

# The rule's deadlock case: three low-battery drones whose only base-ward cells collide.
deadlock = WildfireFleetEnv(EnvConfig(**development['environment']))
deadlock.reset(seed=5000)
deadlock.positions = np.array([[9, 9], [8, 10], [10, 8]], dtype=np.int32)
deadlock.batteries = np.array([3.5, 2.0, 3.5], dtype=np.float32)
print('candidate_actions alone:', candidate_actions(deadlock)[0].tolist(), '(all stay → hover until empty)')
print('mask with emergency return:', [decode_action(int(a)).tolist()
                                      for a in np.flatnonzero(action_mask(deadlock))])""")

# ---------------------------------------------------------------------------------- 6
cells.append(heading(6, 'Reinforcement learning agents'))
md(edited(10, ('## 4. Three implemented RL methods\n\n', ''),
          ('Source listings below and in the appendix show the actual shared implementation. '
           'Mapping tables resolve line numbers at execution time and fail if an anchor '
           'disappears, avoiding stale hard-coded line references.',
           'The code cells below are the implementation. Comments in the training loop carry '
           'the pseudocode step labels (D1–D8, R1–R7, A1–A6) so each step can be traced.')))
code(extract('src/wild_drone/learning.py', ['TrainConfig']))
code(extract('src/wild_drone/learning.py', ['tensor_observations', 'FleetNetwork']))
code("""network = FleetNetwork(40, critic=True)
maps, vector = tensor_observations(WildfireFleetEnv(main_env).reset(seed=0)[0])
with torch.no_grad():
    scores, values = network(maps, vector)
assert maps.shape == (1, 9, 40, 40) and vector.shape == (1, 24) and scores.shape == (1, 125)
print('Input/output shapes:', tuple(maps.shape), tuple(vector.shape), tuple(scores.shape))
print('40×40 actor + critic parameters:', sum(p.numel() for p in network.parameters()))""")
md(original(12).replace('### 4.1 DQN', '### 6.1 DQN'))
code(extract('src/wild_drone/learning.py', ['td_targets']))
md(original(14).replace('### 4.2 REINFORCE', '### 6.2 REINFORCE'))
code(extract('src/wild_drone/learning.py',
             ['discounted_returns', 'discounted_policy_loss', 'monte_carlo_terms']))
md(original(16).replace('### 4.3 Synchronous A2C', '### 6.3 Synchronous A2C'))
md("""### 6.4 Shared training loop

One function trains all three agents; the algorithm only changes action selection and the
update rule. It is the training loop that produced the archived results, without the file
saving, resume and audit code of the repository version (`src/wild_drone/masked_learning.py`).
Section 7 checks that both give identical weights. The main study used standard DQN and no
reward shaping, so those optional branches are omitted.""")
code('''def train_masked(algorithm, env_config, t, verbose=True):
    """Train one masked learner from scratch; returns (model, per-episode rows)."""
    assert algorithm in ('dqn', 'reinforce', 'a2c')
    assert not t.double_dqn and t.coordination_shaping_weight == 0, 'not used by the main study'
    random.seed(t.seed)
    np.random.seed(t.seed)
    torch.manual_seed(t.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(t.seed)
    env = WildfireFleetEnv(env_config)
    # D1 / R1 / A1: network (REINFORCE and A2C add a value head), Adam, DQN target + replay
    model_config = {'critic': algorithm == 'a2c' or (
        algorithm == 'reinforce' and t.monte_carlo_baseline),
        'spatial_encoding': t.spatial_encoding}
    model = FleetNetwork(env_config.grid_size, **model_config)
    optimiser = torch.optim.Adam(model.parameters(), lr=t.learning_rate)
    target = None
    if algorithm == 'dqn':
        target = FleetNetwork(env_config.grid_size, spatial_encoding=t.spatial_encoding)
        target.load_state_dict(model.state_dict())
        target.eval()
    replay = deque(maxlen=t.replay_size)
    rows, updates, steps, last_loss = [], 0, 0, None

    def tensors(observations):
        return tensor_observations(observations, spatial_encoding=t.spatial_encoding)

    def optimise(loss):  # D6 / R6 / A5: clipped gradient step with Adam
        nonlocal last_loss, updates
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite training loss')
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), t.max_grad_norm)
        optimiser.step()
        last_loss, updates = float(loss.detach()), updates+1

    for episode in range(t.episodes):
        # D2 / R2 / A2: reset the next training world and compute its feasible actions
        observation, _ = env.reset(seed=t.scenario_seed_start+episode)
        mask = action_mask(env)
        logs, values, entropies, rewards = [], [], [], []
        episode_return, fresh_sum, overdue_sum, done = 0.0, 0.0, 0.0, False
        while not done:
            if algorithm == 'dqn':
                # D3: epsilon-greedy over feasible actions only
                epsilon = t.epsilon_start + min(1, steps/t.epsilon_decay_steps)*(
                    t.epsilon_end-t.epsilon_start)
                if rng.random() < epsilon:
                    action = int(rng.choice(np.flatnonzero(mask)))
                else:
                    with torch.no_grad():
                        q, _ = model(*tensors(observation))
                    action = int(apply_mask(q, mask).argmax(dim=1).item())
            else:
                # R2-R3 / A2: sample from the masked categorical policy; keep log-prob, entropy, value
                logits, value = model(*tensors(observation))
                distribution = Categorical(logits=apply_mask(logits, mask))
                chosen = distribution.sample()
                action = int(chosen.item())
                logs.append(distribution.log_prob(chosen).squeeze())
                entropies.append(distribution.entropy().squeeze())
                if value is not None:
                    values.append(value.squeeze())
            next_observation, reward, terminal, truncated, info = env.step(action)
            done = terminal or truncated
            # Terminal next states are never bootstrapped, so their mask is irrelevant.
            next_mask = np.ones(125, dtype=bool) if done else action_mask(env)
            steps += 1
            episode_return += reward
            fresh_sum += info['weighted_freshness']
            overdue_sum += info['overdue_fraction']
            rewards.append(reward)
            if algorithm == 'dqn':
                # D4: store the transition with the next state's mask
                replay.append((observation, action, reward, next_observation, terminal, next_mask))
                if steps >= t.warmup_steps and len(replay) >= t.batch_size:
                    # D5: minibatch and detached TD target over feasible next actions
                    batch = [replay[int(i)] for i in rng.choice(len(replay), t.batch_size,
                                                                 replace=False)]
                    q, _ = model(*tensors([b[0] for b in batch]))
                    actions = torch.tensor([b[1] for b in batch])
                    selected = q.gather(1, actions[:, None]).squeeze(1)
                    with torch.no_grad():
                        next_q, _ = target(*tensors([b[3] for b in batch]))
                        targets = td_targets(torch.tensor([b[2] for b in batch]),
                                             torch.tensor([b[4] for b in batch]),
                                             masked_max(next_q, np.stack([b[5] for b in batch])),
                                             t.gamma)
                    optimise(nn.functional.smooth_l1_loss(selected, targets))  # D6: Huber loss
                if steps % t.target_update_steps == 0:  # D7: copy weights to the target network
                    target.load_state_dict(model.state_dict())
            elif algorithm == 'a2c' and (len(rewards) >= t.n_steps or done):
                # A3: bootstrap from the critic (zero at terminal)
                with torch.no_grad():
                    _, boot = model(*tensors(next_observation))
                bootstrap = 0.0 if terminal else float(boot.item())
                # A4: n-step returns and advantages; A5: actor + critic - entropy
                returns = torch.from_numpy(discounted_returns(rewards, t.gamma, bootstrap))
                predictions = torch.stack(values)
                policy_loss = discounted_policy_loss(
                    torch.stack(logs), returns-predictions.detach(), t.gamma,
                    start_step=env.elapsed-len(rewards), normalizer=t.n_steps)
                optimise(policy_loss + t.value_weight*nn.functional.mse_loss(predictions, returns)
                         - t.entropy_weight*torch.stack(entropies).mean())
                logs, values, entropies, rewards = [], [], [], []  # A6: clear the segment
            observation, mask = next_observation, next_mask
        if algorithm == 'reinforce':
            # R4-R6: full-episode reward-to-go, baseline-corrected actor loss, one update
            actor, critic = monte_carlo_terms(
                torch.stack(logs), rewards, t.gamma, horizon=env_config.horizon,
                values=torch.stack(values) if t.monte_carlo_baseline else None)
            optimise(actor + t.value_weight*critic - t.entropy_weight*torch.stack(entropies).mean())
        completed = episode+1
        rows.append({'episode': completed, 'scenario_seed': t.scenario_seed_start+episode,
                     'training_seed': t.seed, 'return': episode_return, 'steps': env.elapsed,
                     'loss': last_loss, 'updates': updates, 'total_losses': int(env.lost.sum()),
                     'weighted_freshness': fresh_sum/env_config.horizon,
                     'overdue_fraction': (overdue_sum+env_config.horizon-env.elapsed)/
                     env_config.horizon, **env.totals})
        if verbose and t.progress_interval and completed % t.progress_interval == 0:
            print(f'{algorithm}: episode {completed}/{t.episodes}, return {episode_return:.3f}')
    return model, rows  # D8 / R7 / A6: final weights (no validation-based selection)''')
md("""### 6.5 Evaluation

Learned policies act greedily: the highest-scoring feasible action (Q value for DQN, logit
for the actors). `run_episode` flies one complete mission and reports the metrics used in
every table. If the whole fleet is lost early, the rest of the mission counts as zero
freshness and fully overdue, so failing early is never rewarded.""")
code('''class MaskedNeuralPolicy:
    """Deterministic evaluation: argmax over feasible actions."""

    def __init__(self, model):
        self.model = model.eval()

    def act(self, env, observation):
        with torch.no_grad():
            logits, _ = self.model(*tensor_observations(
                observation, spatial_encoding=self.model.spatial_encoding))
        return int(apply_mask(logits, action_mask(env)).argmax(dim=1).item())


def run_episode(config, policy, scenario_seed, *, name, training_seed=None):
    env = WildfireFleetEnv(config)
    observation, _ = env.reset(seed=scenario_seed)
    fresh, overdue, decisions = [], [], []
    done, total = False, 0.0
    while not done:
        started = time.perf_counter()
        action = policy.act(env, observation)
        decisions.append((time.perf_counter()-started)*1000)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total += reward
        fresh.append(info['weighted_freshness'])
        overdue.append(info['overdue_fraction'])
    missing = config.horizon-len(fresh)  # unserved remainder after a total fleet loss
    return {'policy': name, 'training_seed': training_seed, 'scenario_seed': scenario_seed,
            'grid_size': config.grid_size, 'horizon': config.horizon, 'steps': env.elapsed,
            'return': total, 'weighted_freshness': float(sum(fresh)/config.horizon),
            'overdue_fraction': float((sum(overdue)+missing)/config.horizon),
            'energy_used': env.totals['energy_used'], 'total_losses': int(env.lost.sum()),
            'safe_returns': env.totals['safe_returns'],
            'charge_completions': env.totals['charge_completions'],
            'collision_attempts': env.totals['collision_attempts'],
            'decision_ms': float(np.mean(decisions))}''')

# ---------------------------------------------------------------------------------- 7
cells.append(heading(7, 'Implementation check and demonstration run', """The archived results
were produced by the repository package `src/wild_drone`. When that package is present, this
check runs the notebook's environment, mask and Greedy controller side by side with the
package on the same worlds, and trains each algorithm for two short episodes with both
implementations: trajectories and final weights must be identical. Without the package the
check is skipped and the notebook still runs."""))
code('''small_env = EnvConfig(grid_size=10, horizon=30, battery_capacity=14, recharge_rate=4)
small_train = dict(episodes=2, seed=3, batch_size=8, warmup_steps=8, target_update_steps=8,
                   n_steps=4, progress_interval=0)

if (REPO/'src/wild_drone/masked_learning.py').exists():
    sys.path.insert(0, str(REPO/'src'))
    import wild_drone.baselines as pkg_baselines
    import wild_drone.config as pkg_config
    import wild_drone.environment as pkg_environment
    import wild_drone.learning as pkg_learning
    import wild_drone.masked_learning as pkg_masked

    for world in range(3):
        mine = WildfireFleetEnv(small_env)
        theirs = pkg_environment.WildfireFleetEnv(pkg_config.EnvConfig(**small_env.to_dict()))
        obs_a, _ = mine.reset(seed=world)
        obs_b, _ = theirs.reset(seed=world)
        chooser = np.random.default_rng(world)
        done = False
        while not done:
            assert np.array_equal(action_mask(mine), pkg_masked.action_mask(theirs))
            assert GreedyPolicy().act(mine, obs_a) == pkg_baselines.GreedyPolicy().act(theirs, obs_b)
            action = int(chooser.choice(np.flatnonzero(action_mask(mine))))
            obs_a, reward_a, done, _, info_a = mine.step(action)
            obs_b, reward_b, done_b, _, info_b = theirs.step(action)
            assert reward_a == reward_b and done == done_b
            assert all(np.array_equal(obs_a[k], obs_b[k]) for k in obs_a)
    print('Environment, mask and Greedy: identical on 3 worlds.')

    for algorithm in ALGORITHMS:
        model, rows = train_masked(algorithm, small_env, TrainConfig(**small_train), verbose=False)
        with tempfile.TemporaryDirectory() as folder:
            pkg_masked.train_masked(algorithm, pkg_config.EnvConfig(**small_env.to_dict()),
                                    pkg_learning.TrainConfig(**small_train), folder)
            saved = torch.load(Path(folder)/'policy.pt', weights_only=True)['state_dict']
        assert all(torch.equal(model.state_dict()[k], saved[k]) for k in saved)
        print(f'{algorithm}: notebook and package training give identical weights.')
else:
    print('Repository package not found: equivalence check skipped.')''')
code('''# Demonstration: a short DQN run on a small grid (not an experiment).
demo_model, demo_rows = train_masked('dqn', EnvConfig(grid_size=10, horizon=40),
                                     TrainConfig(episodes=10, warmup_steps=64,
                                                 progress_interval=5))
print(pd.DataFrame(demo_rows)[['episode', 'return', 'total_losses', 'updates']].tail(3))''')

# ---------------------------------------------------------------------------------- 8
cells.append(heading(8, 'Development at 20×20: from unmasked to masked agents'))
md("""We developed on a smaller 20×20 configuration before the main 40×40 study. The table
summarises the experiments; each links to its full archived report.

| Stage (dates) | Question | Outcome | Records |
| --- | --- | --- | --- |
| Unmasked RL, 1,000 episodes on a T4 GPU (3 Oct) | Does longer training make the original agents learn? | No. All three lost most of the fleet and scored near random; DQN's Q values exploded (max ≈ 74,000 against a true bound ≈ 80). | `docs/colab1000-results.md` |
| DQN stability and exploration (3 Oct) | Is DQN instability or exploration the cause? | Lower learning rate removed the Q blow-up; Double DQN and slower exploration did not stop fleet loss. | `dqn-rate-probe`, `dqn-stability`, `exploration` |
| Supervised imitation of Greedy (3–4 Oct) | Can the network represent a good controller at all? | It fitted 86–96% of training decisions but only 9–18% on new worlds; autonomous flights still lost the fleet. Conditional action heads, more data, more updates, recovery data and extra inputs did not fix this. | `imitation`, `action`, `coverage`, `fitting-budget`, `recovery`, `representation` |
| **Feasibility mask (4–5 Oct)** | Is battery safety the missing piece? | **Yes for survival:** zero losses for all three agents; inspection stayed at masked-random level. Results below. | `final-2026-10-05` |
| Masked Double DQN, coordination shaping (10 Oct) | Do a corrected target or a shaped reward add inspection skill? | No gain over masked random; shaping was worse than unshaped. | `double-dqn`, `coordination` |
| Inspection-scoring diagnostics (10 Oct) | Can a network rank inspection actions on new worlds? | Fitting was easy but transfer weak; drone-centred inputs and 5× more worlds helped, reaching near-Greedy missions at 20×20 — with supervised targets, not RL. | `inspection-targets`, `rankings`, `centered-inspection`, `training-coverage`, `mission-validation` |
| 40×40 supervised pilot (11 Oct) | Does that scorer scale to 40×40? | Freshness 0.19200 versus Greedy 0.227 and masked random 0.087; a ranking loss did worse (0.13559). Not RL. | `scale-pilot`, `scale-reference`, `scale-loss` |

Records are under `docs/experiments/` unless shown otherwise. Before the main study, six
correctness fixes (checkpoint loading and study-integrity checks) were made and verified
(`docs/verification-2026-10-10.md`); the 20×20 results below predate them.""")
md(edited(45, ('## 6. Historical 20×20 masked and unrestricted comparison\n\n', ''),
          ('Below are the complete archived test and tuning tables, learning curves for the nine '
           'masked fits and nine unrestricted test fits, and the sensitivity analysis. Tuning is a '
           'one-seed 200-episode development screen with final and best validation outcomes '
           'retained. It does not estimate robust hyperparameter effects. Frozen main DQN uses a '
           'different declared learning rate; the complete current settings are displayed in '
           'Section 7.',
           'Below are the archived test table and the learning curves of the nine masked and nine '
           'unrestricted fits. The tuning screen is in Section 9.')))
code("""historical_summary = pd.read_csv(HISTORICAL/'test_summary.csv')
display(historical_summary[['policy', 'return_mean', 'return_sd', 'weighted_freshness_mean',
                            'overdue_fraction_mean', 'total_losses_mean',
                            'charge_completions_mean']])""")
code(original(47))
code(original(48))

# ---------------------------------------------------------------------------------- 9
cells.append(heading(9, 'Hyperparameter tuning', """We tuned in two stages. At 20×20 a
one-seed screen varied each agent's main hyperparameters (200 episodes, five validation
worlds). At 40×40 a learning-rate sweep trained each agent at three rates, 3× either side of
the main-study rate (one seed, 300 episodes, 20 separate validation worlds 23000–23019)."""))
md('### 9.1 20×20 development screen')
code("""tuning = pd.read_csv(HISTORICAL/'tuning.csv')
display(tuning)""")
code(original(49))
md('### 9.2 40×40 learning-rate sweep')
code('''def learning_rate_sweep(algorithm, factors=(1/3, 1, 3), worlds=range(23000, 23020), seed=0):
    """Train one fit per learning rate with every other setting frozen, then evaluate."""
    base = TrainConfig(**frozen['training'][algorithm], seed=seed)
    rows = []
    for factor in factors:
        rate = float(f'{base.learning_rate*factor:.0e}')
        model, _ = train_masked(algorithm, main_env,
                                replace(base, learning_rate=rate, progress_interval=0),
                                verbose=False)
        policy = MaskedNeuralPolicy(model)
        rows += [dict(learning_rate=rate, **run_episode(main_env, policy, w, name=algorithm,
                                                        training_seed=seed)) for w in worlds]
    return pd.DataFrame(rows)


if RUN_SWEEP:
    rerun = pd.concat([learning_rate_sweep(a) for a in ALGORITHMS])
    display(rerun.groupby(['policy', 'learning_rate'])[
        ['return', 'weighted_freshness', 'overdue_fraction']].mean())

sweep = pd.read_csv(FOLLOW_UP/'lr_sweep_summary.csv')
display(sweep[['policy', 'learning_rate', 'frozen_rate', 'weighted_freshness_mean',
               'overdue_fraction_mean', 'return_mean', 'total_losses_mean']])''')
code('''refs = sweep[sweep['learning_rate'].isna()].set_index('policy')
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, metric, label in zip(axes, ['weighted_freshness_mean', 'return_mean'],
                             ['Freshness ↑', 'Return ↑']):
    for algorithm in ALGORITHMS:
        rows = sweep[sweep['policy'] == algorithm].sort_values('learning_rate')
        ax.plot(rows['learning_rate'], rows[metric], marker='o', label=algorithm.upper())
        frozen_row = rows[rows['frozen_rate'] == True]  # noqa: E712
        ax.scatter(frozen_row['learning_rate'], frozen_row[metric], s=90, facecolors='none',
                   edgecolors='black')
    for name, style in [('greedy', '--'), ('masked_random', ':')]:
        ax.axhline(refs.loc[name, metric], color='grey', linestyle=style, label=name)
    ax.set_xscale('log'); ax.set_xlabel('Learning rate (log scale)'); ax.set_title(label)
    ax.grid(alpha=0.2)
axes[0].legend(fontsize=7)
fig.suptitle('40×40 learning-rate sweep — one seed, 20 validation worlds; circles mark the main-study rate')
finish_plot(fig)''')
md("""**Effect of tuning.** No learning rate closed the gap to Greedy: the best fit (DQN at its
main-study rate) reached 0.098 freshness against Greedy's 0.226. REINFORCE improved with a
larger rate and A2C was flat over a 10-fold range. With one seed per setting, differences of
about 0.01 are within the seed-to-seed spread of the main study, so the ordering of rates is
only suggestive. Tuning the learning rate alone did not change the conclusion.""")

# ---------------------------------------------------------------------------------- 10
cells.append(heading(10, 'Main 40×40 study'))
md(edited(52, ('## 7. Frozen 40×40 main study — design and training evidence\n\n', ''),
          ('The user froze the corrected implementation, complete settings and budget after '
           'stopping further loss tuning.',
           'The corrected implementation, all settings and the budget were fixed before any '
           'main-study fit started.')))
code(original(53))
code("""if RUN_TRAINING:  # retrain the frozen DQN seed-0 fit and compare with its archived curve
    model, rows = train_masked('dqn', main_env, TrainConfig(**frozen['training']['dqn'], seed=0))
    archived = pd.read_csv(MAIN/'fits/dqn/seed-0/training.csv')
    print('Max return difference:', np.abs(pd.DataFrame(rows)['return']-archived['return']).max())""")
md('### Training curves')
code(original(54))
md(original(55))
md(original(56).replace('## 8. Fresh-world final evaluation, uncertainty and audit',
                        '### Final evaluation on 20 fresh worlds'))
code(original(57))
code(original(58))
code('''# Key comparisons quoted in the text
fresh = main_means['weighted_freshness']
masked = main_means.loc['masked_random']
with pd.option_context('display.precision', 6):
    display(pd.DataFrame({
        'Greedy freshness ÷ learner': fresh['greedy']/fresh[ALGORITHMS],
        'learner − masked random (freshness)': fresh[ALGORITHMS]-masked['weighted_freshness'],
        'learner − masked random (return)': main_means.loc[ALGORITHMS, 'return']-masked['return'],
    }))''')
md("""**Planner.** A follow-up evaluated the rolling-horizon planner (4-step lookahead,
8 candidates, 2 sampled futures) on the same 20 worlds. It matches Greedy's freshness at about
60× the decision time, so lookahead adds nothing over the one-step rule here.""")
code("""planner = pd.read_csv(FOLLOW_UP/'planner_vs_frozen_table.csv')
display(planner[['policy', 'source', 'weighted_freshness_mean', 'weighted_freshness_sd',
                 'overdue_fraction_mean', 'return_mean', 'total_losses_mean', 'decision_ms_mean']])""")
md('### Performance gates and survival')
code(original(59))
md(original(60))
code(original(61))
md("""An independent replay of all 75,911 evaluation steps reproduced every action, feasible
mask, state hash and reward, with zero drift in network scores. The trained weights are kept
as GitHub Actions artifacts with 90-day retention.""")

# ---------------------------------------------------------------------------------- 11
cells.append(heading(11, 'Sensitivity to battery, charging and fire spread', """The nine
trained learners and the baselines were flown again on the same 20 worlds with one setting
changed at a time (no retraining). Learner rows therefore measure robustness of fixed weights,
not what an agent would learn under the new setting."""))
code('''sensitivity = pd.read_csv(FOLLOW_UP/'sensitivity_summary.csv')
order = ['default', 'battery_capacity=120', 'battery_capacity=200', 'recharge_rate=5',
         'recharge_rate=12', 'spread_probability=0.02', 'spread_probability=0.08']
columns = ['greedy', 'planner', 'masked_random', 'sweep'] + ALGORITHMS
for metric in ['weighted_freshness_mean', 'return_mean']:
    table = sensitivity.pivot(index='condition', columns='policy', values=metric)
    display(table.loc[order, [c for c in columns if c in table.columns]].rename_axis(metric))

per_mission = pd.read_csv(FOLLOW_UP/'sensitivity_episodes.csv')
failures = per_mission.groupby(['condition', 'policy'])['fleet_failure'].sum().unstack(fill_value=0)
display(failures.loc[:, failures.sum() > 0].rename_axis('whole-fleet losses (of 20 or 60)'))''')
code('''table = sensitivity.pivot(index='condition', columns='policy', values='weighted_freshness_mean')
table = table.loc[order, [c for c in columns if c in table.columns]]
fig, ax = plt.subplots(figsize=(11, 3.8))
table.plot.bar(ax=ax, width=0.85)
ax.set_ylabel('Weighted freshness ↑'); ax.set_xlabel('')
ax.set_title('Sensitivity at 40×40 — one setting changed at a time')
ax.grid(axis='y', alpha=0.2); ax.set_ylim(0, 0.29)
ax.legend(fontsize=7, ncol=7, loc='upper center')
plt.xticks(rotation=20)
finish_plot(fig)''')
md("""Greedy and the planner stay at 0.20–0.23 freshness in every condition; the learners,
masked random and sweep stay at 0.06–0.10. Away from the default settings Greedy lost its
whole fleet in 4 of 140 missions: three low-battery drones met near the base, every move
home collided, its rule fell back to "all stay", and they hovered until empty. This is the
deadlock shown in Section 5; the masked learners avoid it through the emergency return.""")

# ---------------------------------------------------------------------------------- 12
cells.append(heading(12, 'Critical review'))
md(original(63).replace('## 9. 20×20 versus 40×40 — clear historical comparison',
                        '### 20×20 versus 40×40'))
code(original(64))
md(strip_heading(original(65)))

# ---------------------------------------------------------------------------------- 13
cells.append(heading(13, 'References'))
refs = re.sub(r'\A\*Preserved dated record[^\n]*\n+## References and their role\n+', '', original(71))
md(refs.rstrip() + """
9. Alshiekh, Bloem, Ehlers, Könighofer, Niekum and Topcu (2018). Safe reinforcement learning
   via shielding. AAAI. Basis for restricting the learned policy to feasible actions.""")

notebook = new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'}})
nbformat.write(notebook, OUT)
print(f'wrote {OUT.relative_to(ROOT)}: {len(cells)} cells')
