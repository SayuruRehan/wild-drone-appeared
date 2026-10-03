# One-fit DQN learning-rate screen

This is a validation-only follow-up to the T4 failure diagnosis, not final held-out
evidence or a three-fit comparison. See [the report](../../colab1000-results.md).
Each rate has one network seed (0), 200 episodes, training scenarios 0–199 and validation
5000–5004 every 20 episodes. All other environment and learning settings are identical.
Final and selected outcomes are both retained. The code is unchanged DQN, without a mask,
shield, reward change or modified target rule.

`summary.csv` aggregates scenarios; `comparison.csv` retains 30 per-scenario rows.
Each rate folder retains training/validation CSVs and complete config/runtime metadata.
`manifest.json` identifies the unchanged learning source by SHA-256. Source metadata records
the notebook/report edits as dirty; CPU PyTorch 2.5.1 and NumPy 2.3.5 differ from the T4 run.
Binary policy/replay files remain local, outside Git.

To reproduce the fits from the repo root, with the RL dependencies installed:

```python
import json
from dataclasses import replace
from pathlib import Path
from wild_drone.config import EnvConfig
from wild_drone.learning import TrainConfig, train

raw = json.loads(Path('configs/development.json').read_text())
env = EnvConfig(**raw['environment'])
settings = TrainConfig(**(raw['training'] | {
    'episodes': 200, 'seed': 0, 'device': 'cpu',
    'validation_seeds': tuple(range(5000, 5005)),
    'checkpoint_interval': 20, 'validation_interval': 20, 'progress_interval': 20,
}))
for rate in [0.0003, 0.0001, 0.00003]:
    train('dqn', env, replace(settings, learning_rate=rate),
          Path('runs/dqn-rate-reproduction') / f'lr-{rate:g}')
```

Use a fresh output directory for a new screen. The best score is the largest five-scenario
mean in `validation.csv`; final is the episode-200 mean. For `max_q` diagnostics, load the
corresponding policy, run deterministic validation trajectories, and record the maximum
of its 125 model outputs at each decision, then the maximum across those decisions/scenarios.
The learning-rate screen does not justify changing the default or running final test seeds.
