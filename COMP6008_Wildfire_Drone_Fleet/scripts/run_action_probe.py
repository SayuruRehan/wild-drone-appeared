"""Joint versus conditional action learning on the full fixed Greedy corpus."""

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from wild_drone.action_probe import (
    ARMS,
    METHOD,
    ActionNetwork,
    fit_action,
    load_action,
)
from wild_drone.baselines import GreedyPolicy
from wild_drone.behavior import behavior_episode
from wild_drone.config import EnvConfig
from wild_drone.coverage_probe import CoverageConfig, collect_rollouts, save_dataset, validate_ids
from wild_drone.training_io import atomic_csv_save, atomic_json_save


def run_study(
    config, training_scenarios, validation_scenarios, fit_seeds, budgets, updates_per_round, out
):
    out = Path(out)
    for ids in (training_scenarios, validation_scenarios, fit_seeds, budgets):
        validate_ids(ids)
    if (
        any(b < 1 for b in budgets)
        or sorted(budgets) != list(budgets)
        or not isinstance(updates_per_round, int)
        or isinstance(updates_per_round, bool)
        or updates_per_round < 1
        or any(b % updates_per_round for b in budgets)
    ):
        raise ValueError("budgets must be increasing positive multiples of updates per round")
    if set(training_scenarios) & set(validation_scenarios):
        raise ValueError("training/validation scenario IDs overlap")
    if out.exists() and any(out.iterdir()):
        raise ValueError("existing action outputs: use a new directory; no resume")
    settings = CoverageConfig(
        rounds=max(budgets) // updates_per_round, updates_per_round=updates_per_round
    )
    torch.set_num_threads(1)
    training, training_audit = collect_rollouts(config, training_scenarios)
    validation, validation_audit = collect_rollouts(config, validation_scenarios)
    out.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[1]
    sources = [
        "scripts/run_action_probe.py",
        "src/wild_drone/action_probe.py",
        "src/wild_drone/coverage_probe.py",
        "src/wild_drone/imitation_probe.py",
        "src/wild_drone/learning.py",
        "src/wild_drone/environment.py",
        "src/wild_drone/baselines.py",
        "src/wild_drone/behavior.py",
        "src/wild_drone/config.py",
        "src/wild_drone/experiments.py",
        "src/wild_drone/training_io.py",
        "docs/action-protocol.md",
    ]
    expected = (len(ARMS) * len(fit_seeds) * (len(budgets) + 1) + 1) * len(validation_scenarios)
    protocol = {
        "study": "joint-versus-conditional-action-learning",
        "method": METHOD,
        "environment": config.to_dict(),
        "coverage_config": asdict(settings),
        "training_scenarios": list(training_scenarios),
        "validation_scenarios": list(validation_scenarios),
        "fit_seeds": list(fit_seeds),
        "budgets": list(budgets),
        "arms": list(ARMS),
        "expected_missions": expected,
        "parameter_counts": {
            arm: sum(p.numel() for p in ActionNetwork(config, arm).parameters()) for arm in ARMS
        },
        "training_data": save_dataset(training, out / "training.pt"),
        "validation_data": save_dataset(validation, out / "validation.pt"),
        "device": "cpu",
        "source_sha256": {
            p: hashlib.sha256((source_root / p).read_bytes()).hexdigest() for p in sources
        },
    }
    atomic_json_save(protocol, out / "protocol.json")
    atomic_csv_save(training_audit, out / "training-collection.csv")
    atomic_csv_save(validation_audit, out / "validation-collection.csv")
    rows, summaries, paired_initial = [], [], {}
    for arm in ARMS:
        for seed in fit_seeds:
            directory = out / arm / f"seed-{seed}"
            started = time.perf_counter()
            summary = fit_action(config, training, validation, settings, seed, arm, directory)
            summaries.append(summary | {"fit_seconds": time.perf_counter() - started})
            manifest = json.loads((directory / "checkpoint_manifest.json").read_text())
            digest = manifest["round-0.pt"]["encoder_sha256"]
            if seed in paired_initial and digest != paired_initial[seed]:
                raise ValueError("matched initial encoder tensors differ across arms")
            paired_initial[seed] = digest
            paths = [f"round-{budget // updates_per_round}.pt" for budget in budgets]
            paths += ["selected_action.pt"]
            for checkpoint in paths:
                record = manifest[checkpoint]
                policy = load_action(directory / checkpoint, record)
                for scenario in validation_scenarios:
                    keep = (
                        arm == ARMS[1]
                        and seed == fit_seeds[0]
                        and checkpoint == f"round-{settings.rounds}.pt"
                        and scenario == validation_scenarios[0]
                    )
                    row, trace = behavior_episode(
                        config, policy, scenario, name=METHOD, training_seed=seed, trace=keep
                    )
                    common = {
                        "arm": arm,
                        "checkpoint": checkpoint,
                        "completed_round": record["completed_round"],
                        "completed_updates": record["completed_updates"],
                        "checkpoint_sha256": record["sha256"],
                    }
                    rows.append(row | common)
                    if keep:
                        atomic_csv_save(
                            [
                                r | common | {"training_seed": seed, "scenario_seed": scenario}
                                for r in trace
                            ],
                            out / "example-trace.csv",
                        )
                atomic_csv_save(rows, out / "validation_missions.csv")
            atomic_json_save(summaries, out / "fits.json")
    for scenario in validation_scenarios:
        row, _ = behavior_episode(config, GreedyPolicy(), scenario, name="greedy")
        rows.append({key: row.get(key) for key in rows[0]})
    atomic_csv_save(rows, out / "validation_missions.csv")
    persisted = list(csv.DictReader((out / "validation_missions.csv").open()))
    if len(persisted) != expected or sum(r["policy"] == "greedy" for r in persisted) != len(
        validation_scenarios
    ):
        raise ValueError("persisted mission count or baseline rows differ from protocol")
    atomic_json_save(
        {
            "persisted_missions": len(persisted),
            "mission_csv_sha256": hashlib.sha256(
                (out / "validation_missions.csv").read_bytes()
            ).hexdigest(),
            "paired_initial_encoder_sha256": paired_initial,
        },
        out / "completion.json",
    )
    print(
        f"Action probe complete: {len(persisted)} persisted missions -> {out}", flush=True
    )
    return protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/development.json"))
    parser.add_argument("--training-scenarios", nargs="+", type=int, default=list(range(100)))
    parser.add_argument(
        "--validation-scenarios", nargs="+", type=int, default=list(range(5000, 5005))
    )
    parser.add_argument("--fit-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budgets", nargs="+", type=int, default=[4000, 8000, 12000, 16000, 20000])
    parser.add_argument("--updates-per-round", type=int, default=4000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    config = EnvConfig(**json.loads(args.config.read_text())["environment"])
    run_study(
        config,
        args.training_scenarios,
        args.validation_scenarios,
        args.fit_seeds,
        args.budgets,
        args.updates_per_round,
        args.out,
    )


if __name__ == "__main__":
    main()
