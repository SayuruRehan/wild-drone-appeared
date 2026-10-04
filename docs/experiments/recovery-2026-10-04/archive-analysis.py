"""Recompute paired aggregation and figure from archived raw records."""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

out = Path(__file__).resolve().parent
root = out / "raw"
protocol = json.loads((root / "protocol.json").read_text())
rows = list(csv.DictReader((root / "validation_missions.csv").open()))
fits = json.loads((root / "fits.json").read_text())
seeds = protocol["fit_seeds"]
arms = protocol["arms"]
budgets = [
    r * protocol["coverage_config"]["updates_per_round"]
    for r in range(1, protocol["coverage_config"]["rounds"] + 1)
]
interval = protocol["coverage_config"]["updates_per_round"]
assert len(rows) == protocol["expected_missions"] == 185
metrics = [
    "return",
    "weighted_freshness",
    "total_losses",
    "safe_returns",
    "charge_completions",
    "wall_commands",
    "collision_attempts",
]
per_fit = []
for arm in arms:
    for budget in budgets + ["selected"]:
        checkpoint = (
            "selected_recovery.pt" if budget == "selected" else f"round-{budget // interval}.pt"
        )
        for seed in seeds:
            group = [
                r
                for r in rows
                if r["training_seed"] == str(seed)
                and r["arm"] == arm
                and r["checkpoint"] == checkpoint
            ]
            assert len(group) == 5
            opportunities = sum(int(r["critical_drone_steps"]) for r in group)
            moves = sum(int(r["critical_return_moves"]) for r in group)
            per_fit.append(
                {
                    "arm": arm,
                    "budget": budget,
                    "training_seed": seed,
                    "checkpoint": checkpoint,
                    **{m: float(np.mean([float(r[m]) for r in group])) for m in metrics},
                    "full_fleet_losses": sum(int(r["total_losses"]) == 3 for r in group),
                    "critical_drone_steps": opportunities,
                    "critical_return_moves": moves,
                    "critical_return_move_fraction": moves / opportunities
                    if opportunities
                    else None,
                }
            )
with (out / "per-fit.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(per_fit[0]))
    w.writeheader()
    w.writerows(per_fit)


def stats(values):
    values = [v for v in values if v is not None]
    return {
        "mean": float(np.mean(values)) if values else None,
        "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "units": len(values),
    }


summary = {}
for arm in arms:
    summary[arm] = {}
    for budget in budgets + ["selected"]:
        group = [r for r in per_fit if r["budget"] == budget and r["arm"] == arm]
        summary[arm][str(budget)] = {
            m: stats([r[m] for r in group]) for m in metrics + ["critical_return_move_fraction"]
        }
        summary[arm][str(budget)]["full_fleet_losses"] = sum(r["full_fleet_losses"] for r in group)
baseline = [r for r in rows if r["policy"] == "greedy"]
assert len(baseline) == 5
summary["greedy"] = {m: stats([float(r[m]) for r in baseline]) for m in metrics}
summary["greedy"]["sd_unit"] = "scenario"
opportunities = sum(int(r["critical_drone_steps"]) for r in baseline)
moves = sum(int(r["critical_return_moves"]) for r in baseline)
summary["greedy"]["critical_return_move_fraction"] = {
    "mean": moves / opportunities if opportunities else None,
    "sample_sd": None,
    "units": 1,
}

fields = [
    "joint_accuracy",
    "live_command_accuracy",
    "critical_command_accuracy",
    "critical_toward_base_command_fraction",
    "charging_hold_fraction",
    "cross_entropy",
]
label_rows = []
for arm in arms:
    for seed in seeds:
        records = list(csv.DictReader((root / arm / f"seed-{seed}/label_metrics.csv").open()))
        assert len(records) == 12
        label_rows.extend([r | {"training_seed": seed, "arm": arm} for r in records])
summary["classification"] = {
    arm: {
        str(b): {
            split: {
                m: stats(
                    [
                        float(r[m])
                        for r in label_rows
                        if int(r["updates"]) == b and r["split"] == split and r["arm"] == arm
                    ]
                )
                for m in fields
            }
            for split in ["train", "validation"]
        }
        for b in budgets
    }
    for arm in arms
}
summary["paired_arm_differences"] = {}
for b in budgets + ["selected"]:
    result = {}
    for metric in metrics + ["critical_return_move_fraction"]:
        differences = []
        for seed in seeds:
            values = {
                r["arm"]: r[metric]
                for r in per_fit
                if r["training_seed"] == seed and r["budget"] == b
            }
            if values[arms[1]] is not None and values[arms[0]] is not None:
                differences.append(values[arms[1]] - values[arms[0]])
        result[metric] = stats(differences)
    summary["paired_arm_differences"][str(b)] = result
summary["selected_rounds"] = {
    arm: [f["selected_round"] for f in fits if f["arm"] == arm] for arm in arms
}
summary["fit_seconds_total"] = sum(f["fit_seconds"] for f in fits)
summary["collection"] = {}
collection_rows = []
for arm in arms:
    summary["collection"][arm] = {}
    for seed in seeds:
        records = json.loads((root / arm / f"seed-{seed}/collections.json").read_text())
        assert len(records) == 4
        totals = {
            m: sum(r[m] for r in records)
            for m in [
                "states",
                "critical_drone_states",
                "charging_drone_states",
                "teacher_executed_steps",
                "unreachable_drone_states",
                "live_drone_states",
                "safe_returns",
                "charge_completions",
                "losses",
            ]
        }
        totals["teacher_execution_fraction"] = totals["teacher_executed_steps"] / totals["states"]
        totals["unreachable_live_fraction"] = (
            totals["unreachable_drone_states"] / totals["live_drone_states"]
        )
        collection_rows.append(totals | {"arm": arm, "seed": seed})
    summary["collection"][arm] = {
        m: stats([r[m] for r in collection_rows if r["arm"] == arm]) for m in totals
    }
with (out / "collection-per-fit.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(collection_rows[0]))
    w.writeheader()
    w.writerows(collection_rows)
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["svg.hashsalt"] = "matched-recovery-2026-10-04"
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
for arm, color in zip(arms, ["#2166ac", "#b35806"]):
    for split, style in [("train", "--"), ("validation", "-")]:
        values = np.array(
            [
                [
                    float(r["joint_accuracy"]) * 100
                    for r in label_rows
                    if r["training_seed"] == seed and r["split"] == split and r["arm"] == arm
                ]
                for seed in seeds
            ]
        )
        x = np.arange(protocol["coverage_config"]["rounds"] + 1) * interval
        mean, sd = values.mean(0), values.std(0, ddof=1)
        axes[0].plot(x, mean, color=color, linestyle=style, label=f"{arm}, {split}")
        axes[0].fill_between(x, mean - sd, mean + sd, color=color, alpha=0.10)
    values = np.array(
        [
            [
                r["return"]
                for b in budgets
                for r in per_fit
                if r["budget"] == b and r["training_seed"] == seed and r["arm"] == arm
            ]
            for seed in seeds
        ]
    )
    mean, sd = values.mean(0), values.std(0, ddof=1)
    axes[1].errorbar(budgets, mean, yerr=sd, color=color, marker="o", capsize=4, label=arm)
axes[1].axhline(
    summary["greedy"]["return"]["mean"], color="#555555", linestyle="--", label="Greedy mean"
)
axes[0].set_title("Recovery collection decision fitting")
axes[0].set_ylabel("Joint-action agreement (%)")
axes[0].set_ylim(0, 100)
axes[1].set_title("Autonomous validation flight")
axes[1].set_ylabel("Mission return (higher is better)")
for ax in axes:
    ax.set_xlabel("Optimizer updates")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
fig.suptitle("Mean ± sample SD over three fits per arm; five missions per fit")
fig.savefig(out / "comparison.svg", metadata={"Date": None})
plt.close(fig)
