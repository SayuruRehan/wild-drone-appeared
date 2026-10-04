"""Supervised joint versus conditional action-learning diagnostic; not RL."""

import copy
import hashlib
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from wild_drone.config import EnvConfig
from wild_drone.coverage_probe import model_digest, validate_ids
from wild_drone.experiments import write_metadata
from wild_drone.imitation_probe import (
    ProbePolicy,
    _export,
    _validate_data,
    data_digest,
    label_metrics,
)
from wild_drone.learning import FleetNetwork
from wild_drone.training_io import atomic_csv_save, atomic_json_save

FORMAT = "wild-drone-conditional-action-imitation-v1"
METHOD = "supervised-conditional-action-probe"
ARMS = ("joint", "conditional")
MODEL_CONFIG = {"conditional_width": 64, "order": [0, 1, 2], "decoding": "joint MAP"}


def encoder_digest(state):
    return model_digest({k: v for k, v in state.items() if k.startswith(("conv.", "encoder."))})


class ActionNetwork(FleetNetwork):
    def __init__(self, config, arm):
        if arm not in ARMS:
            raise ValueError("invalid action arm")
        super().__init__(config.grid_size)
        self.arm = arm
        if arm == "conditional":
            del self.head
            self.first = nn.Linear(128, 5)
            self.second = nn.Sequential(nn.Linear(133, 64), nn.ReLU(), nn.Linear(64, 5))
            self.third = nn.Sequential(nn.Linear(138, 64), nn.ReLU(), nn.Linear(64, 5))

    def features(self, maps, vector):
        return self.encoder(torch.cat((self.conv(maps), vector), 1))

    def supervised_loss(self, maps, vector, actions):
        if self.arm == "joint":
            return nn.functional.cross_entropy(self(maps, vector)[0], actions)
        features = self.features(maps, vector)
        first, second, third = actions // 25, (actions // 5) % 5, actions % 5
        a = nn.functional.one_hot(first, 5).to(features.dtype)
        b = nn.functional.one_hot(second, 5).to(features.dtype)
        # The sum is complete joint-action NLL, not the mean of three losses.
        return (
            nn.functional.cross_entropy(self.first(features), first)
            + nn.functional.cross_entropy(self.second(torch.cat((features, a), 1)), second)
            + nn.functional.cross_entropy(self.third(torch.cat((features, a, b), 1)), third)
        )

    def forward(self, maps, vector):
        if self.arm == "joint":
            return super().forward(maps, vector)
        features = self.features(maps, vector)
        batch = len(features)
        commands = torch.eye(5, device=features.device, dtype=features.dtype)
        first = self.first(features).log_softmax(-1)
        second_context = torch.cat(
            (features[:, None].expand(-1, 5, -1), commands[None].expand(batch, -1, -1)), -1
        )
        second = self.second(second_context).log_softmax(-1)
        third_context = torch.cat(
            (
                features[:, None, None].expand(-1, 5, 5, -1),
                commands[None, :, None].expand(batch, -1, 5, -1),
                commands[None, None].expand(batch, 5, -1, -1),
            ),
            -1,
        )
        third = self.third(third_context).log_softmax(-1)
        # Exact joint MAP among all 125 proposals; no per-drone greedy decoding.
        joint = first[:, :, None, None] + second[:, :, :, None] + third
        return joint.reshape(batch, 125), None


def load_action(path, record=None):
    path = Path(path)
    if record and hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("action checkpoint bytes differ from verified manifest")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (
        saved.get("format") != FORMAT
        or saved.get("method") != METHOD
        or saved.get("arm") not in ARMS
        or saved.get("model_config") != MODEL_CONFIG
    ):
        raise ValueError("requires explicitly labeled supervised action checkpoint")
    if record and (
        saved["completed_round"] != record["completed_round"]
        or saved["completed_updates"] != record["completed_updates"]
        or model_digest(saved["state_dict"]) != record["model_sha256"]
    ):
        raise ValueError("action checkpoint identity differs from verified manifest")
    config = EnvConfig(**saved["env_config"])
    model = ActionNetwork(config, saved["arm"])
    model.load_state_dict(saved["state_dict"])
    return ProbePolicy(model, config)


def fit_action(config, training, validation, settings, seed, arm, output_dir):
    for data in (training, validation):
        _validate_data(data, config)
    validate_ids([seed])
    if arm not in ARMS:
        raise ValueError("invalid action arm")
    train_ids, val_ids = [d["scenario_seeds"].unique().tolist() for d in (training, validation)]
    if set(train_ids) & set(val_ids):
        raise ValueError("training/validation scenario IDs overlap")
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("existing action outputs: use a new directory; no resume")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = ActionNetwork(config, arm)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    sampler = torch.Generator().manual_seed(seed)
    train_hash, val_hash = data_digest(training), data_digest(validation)
    metadata = write_metadata(
        out,
        config,
        method=METHOD,
        arm=arm,
        seed=seed,
        coverage_config=asdict(settings),
        model_config=MODEL_CONFIG,
        device="cpu",
        training_data_sha256=train_hash,
        validation_data_sha256=val_hash,
        training_scenarios=train_ids,
        validation_scenarios=val_ids,
    )
    rows, manifest, best, best_accuracy = [], {}, None, -1.0

    def export(saved, name):
        return _export(saved, out / name, progress_key="completed_updates") | {
            "completed_round": saved["completed_round"],
            "model_sha256": model_digest(saved["state_dict"]),
            "encoder_sha256": encoder_digest(saved["state_dict"]),
        }

    for round_number in range(settings.rounds + 1):
        if round_number:
            model.train()
            for _ in range(settings.updates_per_round):
                ids = torch.randint(
                    len(training["actions"]), (settings.batch_size,), generator=sampler
                )
                loss = model.supervised_loss(
                    training["maps"][ids], training["vector"][ids], training["actions"][ids]
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite action loss")
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), settings.max_grad_norm, error_if_nonfinite=True
                )
                optimizer.step()
        measured = {}
        for split, data in [("train", training), ("validation", validation)]:
            measured[split] = label_metrics(model, data, settings.batch_size)
            rows.append(
                {
                    "round": round_number,
                    "updates": round_number * settings.updates_per_round,
                    "split": split,
                    "states": len(data["actions"]),
                    **measured[split],
                }
            )
        saved = {
            "format": FORMAT,
            "method": METHOD,
            "arm": arm,
            "env_config": config.to_dict(),
            "coverage_config": asdict(settings),
            "seed": seed,
            "source": metadata,
            "completed_round": round_number,
            "completed_updates": round_number * settings.updates_per_round,
            "training_data_semantic_sha256": train_hash,
            "validation_data_semantic_sha256": val_hash,
            "model_config": MODEL_CONFIG,
            "state_dict": copy.deepcopy(model.state_dict()),
        }
        manifest[f"round-{round_number}.pt"] = export(saved, f"round-{round_number}.pt")
        if round_number and measured["validation"]["joint_accuracy"] > best_accuracy:
            best_accuracy = measured["validation"]["joint_accuracy"]
            best = copy.deepcopy(saved)
            best["selection"] = {"metric": "validation joint_accuracy", "value": best_accuracy}
        atomic_csv_save(rows, out / "label_metrics.csv")
        atomic_json_save(manifest, out / "checkpoint_manifest.json")
        print(
            f"{arm} seed {seed}, round {round_number}/{settings.rounds}, "
            f"train {measured['train']['joint_accuracy']:.3f}, "
            f"validation {measured['validation']['joint_accuracy']:.3f}",
            flush=True,
        )
    manifest["final_action.pt"] = export(saved, "final_action.pt")
    manifest["selected_action.pt"] = export(best, "selected_action.pt")
    atomic_json_save(manifest, out / "checkpoint_manifest.json")
    summary = {
        "method": METHOD,
        "arm": arm,
        "seed": seed,
        "coverage_config": asdict(settings),
        "completed_updates": settings.rounds * settings.updates_per_round,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "model_config": MODEL_CONFIG,
        "training_states": len(training["actions"]),
        "training_data_sha256": train_hash,
        "validation_data_sha256": val_hash,
        "selected_round": best["completed_round"],
        "selected_validation_joint_accuracy": best_accuracy,
        "final_train_metrics": measured["train"],
        "final_validation_metrics": measured["validation"],
    }
    atomic_json_save(summary, out / "action_summary.json")
    return summary
