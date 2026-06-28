from __future__ import annotations

import zipfile
from types import SimpleNamespace

import numpy as np
import torch

from optimizer.policy_checkpoint import load_policy_from_checkpoint


def _env_conf():
    return SimpleNamespace(
        problem_seed=0,
        state_space=SimpleNamespace(shape=(3,)),
        action_space=SimpleNamespace(shape=(2,)),
        gym_conf=None,
        env_name="test-env",
    )


def test_load_policy_from_checkpoint_restores_best_actor_state(tmp_path):
    from policies.actor_critic_mlp_policy import ActorCriticMLPPolicy
    from rl.core.actor_state import capture_ppo_actor_snapshot

    base_policy = ActorCriticMLPPolicy(_env_conf(), (4,))
    checkpoint_policy = base_policy.clone()
    x = np.zeros(checkpoint_policy.num_params(), dtype=np.float64)
    x[: checkpoint_policy.actor_backbone[1].weight.numel()] = 0.1
    checkpoint_policy.set_params(x)
    snapshot = capture_ppo_actor_snapshot(
        checkpoint_policy.actor_backbone,
        checkpoint_policy.actor_head,
        log_std=checkpoint_policy.log_std,
    )
    checkpoint_path = tmp_path / "checkpoint_last.pt"
    torch.save({"best_actor_state": snapshot}, checkpoint_path)

    loaded_policy = load_policy_from_checkpoint(base_policy, checkpoint_path)

    for key, expected in checkpoint_policy.actor_backbone.state_dict().items():
        torch.testing.assert_close(loaded_policy.actor_backbone.state_dict()[key], expected)
    for key, expected in checkpoint_policy.actor_head.state_dict().items():
        torch.testing.assert_close(loaded_policy.actor_head.state_dict()[key], expected)
    torch.testing.assert_close(loaded_policy.log_std, checkpoint_policy.log_std)
    np.testing.assert_allclose(loaded_policy.get_params(), np.zeros(loaded_policy.num_params()), atol=0.0, rtol=0.0)


def test_load_policy_from_checkpoint_requires_actor_modules(tmp_path):
    from policies.mlp_policy import MLPPolicy

    checkpoint_path = tmp_path / "checkpoint_last.pt"
    torch.save({"best_actor_state": {"backbone": {}, "head": {}, "log_std": torch.zeros(2)}}, checkpoint_path)

    policy = MLPPolicy(_env_conf(), hidden_sizes=(4,))
    try:
        load_policy_from_checkpoint(policy, checkpoint_path)
    except ValueError as exc:
        assert "actor_backbone" in str(exc)
    else:
        raise AssertionError("Expected actor module validation to fail.")


def test_load_policy_from_sb3_zip_restores_actor_and_rebases_origin(tmp_path):
    from policies.actor_critic_mlp_policy import ActorCriticMLPPolicy

    policy = ActorCriticMLPPolicy(
        _env_conf(),
        (4, 4),
        activation="tanh",
        layer_norm=False,
        share_backbone=False,
        squash_action=False,
        deterministic_call=True,
    )
    sb3_state = {
        "log_std": torch.linspace(-0.5, 0.5, 2),
        "mlp_extractor.policy_net.0.weight": torch.full((4, 3), 0.11),
        "mlp_extractor.policy_net.0.bias": torch.full((4,), 0.12),
        "mlp_extractor.policy_net.2.weight": torch.full((4, 4), 0.13),
        "mlp_extractor.policy_net.2.bias": torch.full((4,), 0.14),
        "action_net.weight": torch.full((2, 4), 0.15),
        "action_net.bias": torch.full((2,), 0.16),
    }
    policy_pth = tmp_path / "policy.pth"
    torch.save(sb3_state, policy_pth)
    zip_path = tmp_path / "sb3_model.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(policy_pth, "policy.pth")

    loaded_policy = load_policy_from_checkpoint(policy, zip_path)

    torch.testing.assert_close(loaded_policy.actor_backbone[0].weight, sb3_state["mlp_extractor.policy_net.0.weight"])
    torch.testing.assert_close(loaded_policy.actor_backbone[2].bias, sb3_state["mlp_extractor.policy_net.2.bias"])
    torch.testing.assert_close(loaded_policy.actor_head.weight, sb3_state["action_net.weight"])
    torch.testing.assert_close(loaded_policy.log_std, sb3_state["log_std"])
    np.testing.assert_allclose(loaded_policy.get_params(), np.zeros(loaded_policy.num_params()), atol=0.0, rtol=0.0)
