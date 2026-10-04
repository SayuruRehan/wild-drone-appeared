"""Post-hoc single-world memorization check; never evidence of generalization."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from wild_drone.baselines import GreedyPolicy
from wild_drone.behavior import behavior_episode
from wild_drone.config import EnvConfig
from wild_drone.coverage_probe import (
    CoverageConfig,
    collect_rollouts,
    fit_coverage,
    load_coverage,
    save_dataset,
)
from wild_drone.training_io import atomic_csv_save, atomic_json_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise ValueError("fresh competence output folder required")
    config = EnvConfig(**json.loads(Path("configs/development.json").read_text())["environment"])
    sources = [
        "src/wild_drone/coverage_probe.py",
        "src/wild_drone/imitation_probe.py",
        "src/wild_drone/learning.py",
        "src/wild_drone/environment.py",
        "src/wild_drone/baselines.py",
        "src/wild_drone/behavior.py",
        "src/wild_drone/config.py",
        "src/wild_drone/training_io.py",
        "src/wild_drone/experiments.py",
    ]
    settings = CoverageConfig(rounds=1, updates_per_round=2000)
    worlds = [0, *range(5000, 5005)]
    out.mkdir(parents=True)
    # Conditions are fixed before collecting or fitting; do not pick a successful world.
    protocol = {
        "study": "post-hoc-single-world-competence-check",
        "environment": config.to_dict(),
        "training_worlds": [0],
        "validation_worlds": worlds[1:],
        "fit_seeds": [0, 1, 2],
        "updates": 2000,
        "batch_size": 64,
        "learning_rate": 0.001,
        "max_grad_norm": 0.5,
        "evaluation": "final checkpoint only; separate trained world0 and five validation worlds",
        "expected_missions": 24,
        "device": "cpu",
        "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    atomic_json_save(protocol, out / "protocol.json")
    torch.set_num_threads(1)
    training, _ = collect_rollouts(config, [0])
    validation, _ = collect_rollouts(config, worlds[1:])
    protocol |= {
        "training_data": save_dataset(training, out / "training.pt"),
        "validation_data": save_dataset(validation, out / "validation.pt"),
    }
    atomic_json_save(protocol, out / "protocol.json")
    rows = []
    for seed in protocol["fit_seeds"]:
        folder = out / f"seed-{seed}"
        fit_coverage(config, training, validation, [], settings, seed, "fixed", folder)
        manifest = json.loads((folder / "checkpoint_manifest.json").read_text())
        record = manifest["final_coverage.pt"]
        policy = load_coverage(folder / "final_coverage.pt", record)
        for world in worlds:
            row, _ = behavior_episode(
                config, policy, world, name="single-world-competence", training_seed=seed
            )
            rows.append(
                row
                | {
                    "split": "training" if world == 0 else "validation",
                    "checkpoint_sha256": record["sha256"],
                }
            )
        atomic_csv_save(rows, out / "missions.csv")
    for world in worlds:
        row, _ = behavior_episode(config, GreedyPolicy(), world, name="greedy")
        rows.append(
            row | {"split": "training" if world == 0 else "validation", "checkpoint_sha256": None}
        )
    atomic_csv_save(rows, out / "missions.csv")
    assert len(rows) == protocol["expected_missions"]
    atomic_json_save(
        {
            "persisted_missions": len(rows),
            "mission_csv_sha256": hashlib.sha256((out / "missions.csv").read_bytes()).hexdigest(),
        },
        out / "completion.json",
    )
    print("Post-hoc competence check complete; keep trained-world results separate.", flush=True)


if __name__ == "__main__":
    main()
