import importlib
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from wild_drone.config import EnvConfig
from wild_drone.coverage_probe import CoverageConfig, fit_coverage
from wild_drone.environment import WildfireFleetEnv
from wild_drone.imitation_probe import collect_teacher_data
from wild_drone.learning import tensor_observations


def module():
    assert importlib.util.find_spec("wild_drone.action_probe"), "conditional action probe missing"
    return importlib.import_module("wild_drone.action_probe")


def test_conditional_joint_probabilities_preserve_action_order_and_global_map():
    probe = module()
    model = probe.ActionNetwork(EnvConfig(grid_size=8, horizon=6), "conditional")
    # A locally most likely first command (0) is not the best complete fleet action.
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.first.bias.copy_(torch.tensor([0.4, 0.35, 0.1, 0.1, 0.05]).log())
        model.second[0].weight[0, 129] = 1  # First command is 1.
        model.second[2].weight[2, 0] = 8
        model.third[0].weight[0, 129] = 1
        model.third[2].weight[3, 0] = 8
    logits, _ = model(torch.zeros(2, 9, 8, 8), torch.zeros(2, 24))
    torch.testing.assert_close(logits.exp().sum(1), torch.ones(2))
    assert logits.argmax(1).tolist() == [38, 38]  # 1*25 + 2*5 + 3.
    assert float(logits[0, 38].detach()) == pytest.approx(
        math.log(0.35) - 2 * math.log1p(4 * math.exp(-8)), abs=1e-6
    )
    assert float(logits[0, 0].detach()) == pytest.approx(math.log(0.4) - 2 * math.log(5), abs=1e-6)


def test_teacher_context_loss_and_gradients_equal_full_joint_likelihood():
    probe = module()
    torch.manual_seed(0)
    model = probe.ActionNetwork(EnvConfig(grid_size=8, horizon=6), "conditional")
    maps, vector = torch.randn(2, 9, 8, 8), torch.randn(2, 24)
    actions = torch.tensor([38, 124])
    loss = model.supervised_loss(maps, vector, actions)
    loss.backward()
    gradients = {k: p.grad.clone() for k, p in model.named_parameters()}
    model.zero_grad()
    full = torch.nn.functional.cross_entropy(model(maps, vector)[0], actions)
    full.backward()
    torch.testing.assert_close(loss, full)
    for k, p in model.named_parameters():
        torch.testing.assert_close(gradients[k], p.grad, atol=2e-6, rtol=2e-5)


def test_control_matches_fixed_fitter_and_both_heads_share_initial_encoder(tmp_path):
    probe = module()
    cfg = EnvConfig(grid_size=8, horizon=6)
    train, val = collect_teacher_data(cfg, [0, 1]), collect_teacher_data(cfg, [5000])
    settings = CoverageConfig(rounds=1, updates_per_round=4)
    fit_coverage(cfg, train, val, [], settings, 0, "fixed", tmp_path / "original")
    for arm in probe.ARMS:
        probe.fit_action(cfg, train, val, settings, 0, arm, tmp_path / arm)
    manifests = {
        name: json.loads((tmp_path / name / "checkpoint_manifest.json").read_text())
        for name in ["original", *probe.ARMS]
    }
    assert (
        manifests["joint"]["round-1.pt"]["model_sha256"]
        == manifests["original"]["round-1.pt"]["model_sha256"]
    )
    assert (
        manifests["joint"]["round-0.pt"]["encoder_sha256"]
        == manifests["conditional"]["round-0.pt"]["encoder_sha256"]
    )


def test_action_exports_validation_isolation_and_unshielded_deployment(tmp_path):
    probe = module()
    cfg = EnvConfig(grid_size=8, horizon=6)
    train, val = collect_teacher_data(cfg, [0, 1]), collect_teacher_data(cfg, [5000])
    altered = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in val.items()}
    altered["actions"] = (altered["actions"] + 1) % 125
    settings = CoverageConfig(rounds=2, updates_per_round=4)
    for name, data in [("a", val), ("b", altered)]:
        probe.fit_action(cfg, train, data, settings, 0, "conditional", tmp_path / name)
    manifests = [
        json.loads((tmp_path / name / "checkpoint_manifest.json").read_text())
        for name in ["a", "b"]
    ]
    assert (
        manifests[0]["final_action.pt"]["model_sha256"]
        == manifests[1]["final_action.pt"]["model_sha256"]
    )
    policy = probe.load_action(tmp_path / "a/final_action.pt", manifests[0]["final_action.pt"])
    env = WildfireFleetEnv(cfg)
    obs, _ = env.reset(seed=5000)
    with torch.no_grad():
        action = int(policy.model(*tensor_observations(obs))[0].argmax(1))
    assert policy.act(env, obs) == action
    from wild_drone.learning import load_policy

    with pytest.raises(KeyError):
        load_policy(tmp_path / "a/final_action.pt")
    with pytest.raises(ValueError, match="identity"):
        probe.load_action(
            tmp_path / "a/final_action.pt",
            manifests[0]["final_action.pt"] | {"model_sha256": "wrong"},
        )
    with pytest.raises(ValueError, match="overlap"):
        probe.fit_action(cfg, train, train, settings, 0, "conditional", tmp_path / "bad")
    with pytest.raises(ValueError, match="existing"):
        probe.fit_action(cfg, train, val, settings, 0, "conditional", tmp_path / "a")
    with pytest.raises(ValueError, match="arm"):
        probe.ActionNetwork(cfg, "unknown")


def test_action_runner_persists_baseline_and_shared_encoder_identity(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/run_action_probe.py"
    assert path.exists(), "conditional action runner missing"
    spec = importlib.util.spec_from_file_location("action_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    out = tmp_path / "run"
    runner.run_study(EnvConfig(grid_size=8, horizon=6), [0, 1], [5000], [0], [4, 8], 4, out)
    complete = json.loads((out / "completion.json").read_text())
    assert complete["persisted_missions"] == 7
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["parameter_counts"]["joint"] != protocol["parameter_counts"]["conditional"]
    with pytest.raises(ValueError, match="existing"):
        runner.run_study(EnvConfig(grid_size=8, horizon=6), [0, 1], [5000], [0], [4, 8], 4, out)
