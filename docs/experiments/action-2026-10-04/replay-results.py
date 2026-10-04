import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

from wild_drone.action_probe import MODEL_CONFIG, encoder_digest, load_action
from wild_drone.baselines import GreedyPolicy
from wild_drone.behavior import behavior_episode
from wild_drone.config import EnvConfig
from wild_drone.coverage_probe import model_digest
from wild_drone.imitation_probe import data_digest, label_metrics

torch.set_num_threads(1)
root = Path(sys.argv[1])
protocol = json.loads((root / "protocol.json").read_text())
config = EnvConfig(**protocol["environment"])
for path, digest in protocol["source_sha256"].items():
    actual = Path(path).read_bytes()
    if hashlib.sha256(actual).hexdigest() != digest and path == "scripts/run_action_probe.py":
        original = (root / "original-runner.py").read_bytes()
        assert (
            actual.replace(b"Action probe complete:", b"Representation probe complete:") == original
        )
        actual = original
    assert hashlib.sha256(actual).hexdigest() == digest, path
for key in ["training_data", "validation_data"]:
    record = protocol[key]
    path = root / record["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    assert data_digest(torch.load(path, weights_only=True)) == record["semantic_sha256"]
data_by_split = {
    key: torch.load(
        root / protocol[f"{key if key == 'validation' else 'training'}_data"]["file"],
        weights_only=True,
    )
    for key in ["train", "validation"]
}
initial = {}
count = 0
label_checks = 0
for arm in protocol["arms"]:
    for seed in protocol["fit_seeds"]:
        folder = root / arm / f"seed-{seed}"
        manifest = json.loads((folder / "checkpoint_manifest.json").read_text())
        records = list(csv.DictReader((folder / "label_metrics.csv").open()))
        selected = max(
            [r for r in records if r["split"] == "validation" and int(r["round"]) > 0],
            key=lambda r: float(r["joint_accuracy"]),
        )
        for name, record in manifest.items():
            path = folder / name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
            saved = torch.load(path, weights_only=True)
            assert saved["completed_updates"] == record["completed_updates"]
            assert saved["completed_round"] == record["completed_round"]
            assert (
                saved["completed_updates"]
                == saved["completed_round"] * protocol["coverage_config"]["updates_per_round"]
            )
            assert saved["arm"] == arm and saved["seed"] == seed
            assert saved["env_config"] == protocol["environment"]
            assert saved["coverage_config"] == protocol["coverage_config"]
            assert saved["model_config"] == MODEL_CONFIG
            assert saved["source"] == json.loads((folder / "metadata.json").read_text())
            assert not saved["source"]["git_dirty"]
            assert encoder_digest(saved["state_dict"]) == record["encoder_sha256"]
            assert (
                sum(v.numel() for v in saved["state_dict"].values())
                == protocol["parameter_counts"][arm]
            )
            assert (
                saved["training_data_semantic_sha256"]
                == protocol["training_data"]["semantic_sha256"]
            )
            assert (
                saved["validation_data_semantic_sha256"]
                == protocol["validation_data"]["semantic_sha256"]
            )
            assert model_digest(saved["state_dict"]) == record["model_sha256"]
            if name == "selected_action.pt":
                assert saved["completed_round"] == int(selected["round"])
                assert (
                    record["model_sha256"]
                    == manifest[f"round-{saved['completed_round']}.pt"]["model_sha256"]
                )
            if name == "final_action.pt":
                assert (
                    record["model_sha256"]
                    == manifest[f"round-{protocol['coverage_config']['rounds']}.pt"]["model_sha256"]
                )
            count += 1
        for round_number in range(protocol["coverage_config"]["rounds"] + 1):
            name = f"round-{round_number}.pt"
            policy = load_action(folder / name, manifest[name])
            for split, data in data_by_split.items():
                old = next(
                    r for r in records if int(r["round"]) == round_number and r["split"] == split
                )
                current = label_metrics(
                    policy.model, data, protocol["coverage_config"]["batch_size"]
                )
                for key, value in current.items():
                    if value is None:
                        assert old[key] == ""
                    else:
                        assert float(old[key]) == value, (arm, seed, round_number, split, key)
                label_checks += 1
        print("Checkpoint/label audit passed", arm, seed, flush=True)
        if seed in initial:
            assert initial[seed] == manifest["round-0.pt"]["encoder_sha256"]
        initial[seed] = manifest["round-0.pt"]["encoder_sha256"]

rows = list(csv.DictReader((root / "validation_missions.csv").open()))
completion = json.loads((root / "completion.json").read_text())
assert len(rows) == protocol["expected_missions"]
assert sum(r["policy"] == "greedy" for r in rows) == len(protocol["validation_scenarios"])
assert (
    hashlib.sha256((root / "validation_missions.csv").read_bytes()).hexdigest()
    == completion["mission_csv_sha256"]
)
policies = {}
compared = 0
trace_checks = 0
for old in rows:
    if old["policy"] == "greedy":
        policy = GreedyPolicy()
        seed = None
    else:
        seed = int(old["training_seed"])
        key = old["arm"], seed, old["checkpoint"]
        folder = root / old["arm"] / f"seed-{seed}"
        if key not in policies:
            record = json.loads((folder / "checkpoint_manifest.json").read_text())[
                old["checkpoint"]
            ]
            assert record["sha256"] == old["checkpoint_sha256"]
            policies[key] = load_action(folder / old["checkpoint"], record)
        policy = policies[key]
    trace = (
        old["arm"] == "conditional"
        and seed == 0
        and old["checkpoint"] == f"round-{protocol['coverage_config']['rounds']}.pt"
        and old["scenario_seed"] == str(protocol["validation_scenarios"][0])
    )
    current, recorded = behavior_episode(
        config,
        policy,
        int(old["scenario_seed"]),
        name=old["policy"],
        training_seed=seed,
        trace=trace,
    )
    for name, value in current.items():
        if name == "decision_ms":
            continue
        if value is None:
            assert old[name] == ""
        elif isinstance(value, (int, float)):
            assert float(old[name]) == value, (old["arm"], seed, old["scenario_seed"], name)
        else:
            assert str(value) == old[name]
        compared += 1
    if trace:
        saved = list(csv.DictReader((root / "example-trace.csv").open()))
        assert len(saved) == len(recorded)
        for a, b in zip(saved, recorded):
            for name, value in b.items():
                if isinstance(value, bool):
                    assert a[name] == str(value)
                else:
                    assert float(a[name]) == value
                trace_checks += 1

training_checks = 0
if (root / "training_missions.csv").exists():
    training_rows = list(csv.DictReader((root / "training_missions.csv").open()))
    assert len(training_rows) == 7
    for old in training_rows:
        if old["policy"] == "greedy":
            policy = GreedyPolicy()
            seed = None
        else:
            seed = int(old["training_seed"])
            folder = root / old["arm"] / f"seed-{seed}"
            record = json.loads((folder / "checkpoint_manifest.json").read_text())[
                "final_action.pt"
            ]
            assert record["sha256"] == old["checkpoint_sha256"]
            policy = load_action(folder / "final_action.pt", record)
        current, _ = behavior_episode(config, policy, 0, name=old["policy"], training_seed=seed)
        for name, value in current.items():
            if name == "decision_ms":
                continue
            if value is None:
                assert old[name] == ""
            elif isinstance(value, (int, float)):
                assert float(old[name]) == value
            else:
                assert str(value) == old[name]
            training_checks += 1
    assert json.loads((root / "sanity-gate.json").read_text())["gate_passed"]

print(
    "Verified source and both datasets,",
    count,
    "checkpoints,",
    len(rows),
    "missions,",
    compared,
    "non-timing fields and",
    trace_checks,
    "example-trace fields,",
    label_checks,
    "recomputed label rows.",
)
(root / "integrity-verification.json").write_text(
    json.dumps(
        {
            "checkpoints": count,
            "missions": len(rows),
            "non_timing_fields": compared,
            "example_trace_fields": trace_checks,
            "datasets": 2,
            "source_files": len(protocol["source_sha256"]),
            "recomputed_label_rows": label_checks,
            "training_mission_fields": training_checks,
        },
        indent=2,
    )
    + "\n"
)
