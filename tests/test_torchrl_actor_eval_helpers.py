from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from rl.algos.backends.torchrl.common.env_contract import ObservationContract
from rl.algos.backends.torchrl.ppo.actor_eval import (
    ActorEvalPolicy,
    capture_actor_snapshot,
    restore_actor_snapshot,
    use_actor_snapshot,
)
from rl.algos.backends.torchrl.sac.actor_eval import (
    SacActorEvalPolicy,
    capture_sac_actor_snapshot,
    restore_sac_actor_snapshot,
)


def _make_actor_modules():
    actor_backbone = nn.Linear(3, 4)
    actor_head = nn.Linear(4, 2)
    obs_scaler = nn.Identity()
    log_std = nn.Parameter(torch.zeros(2))
    return SimpleNamespace(
        actor_backbone=actor_backbone,
        actor_head=actor_head,
        obs_scaler=obs_scaler,
        log_std=log_std,
    )


def test_actor_eval_policy_returns_numpy_action():
    modules = _make_actor_modules()
    policy = ActorEvalPolicy(
        modules.actor_backbone,
        modules.actor_head,
        modules.obs_scaler,
        device=torch.device("cpu"),
        obs_contract=ObservationContract(mode="vector", raw_shape=(3,), vector_dim=3),
    )
    action = policy(np.asarray([0.1, -0.2, 0.3], dtype=np.float32))
    assert isinstance(action, np.ndarray)
    assert action.shape == (2,)


def test_capture_restore_actor_snapshot_roundtrip():
    modules = _make_actor_modules()
    snapshot = capture_actor_snapshot(modules)

    with torch.no_grad():
        modules.actor_backbone.weight.add_(1.0)
        modules.actor_head.bias.sub_(2.0)
        modules.log_std.add_(0.5)

    restore_actor_snapshot(modules, snapshot, device=torch.device("cpu"))
    assert torch.equal(modules.actor_backbone.state_dict()["weight"], snapshot["backbone"]["weight"])
    assert torch.equal(modules.actor_head.state_dict()["bias"], snapshot["head"]["bias"])
    assert torch.allclose(modules.log_std, torch.as_tensor(snapshot["log_std"]))


def test_use_actor_snapshot_temporarily_swaps_state():
    modules = _make_actor_modules()
    original = capture_actor_snapshot(modules)
    replacement = capture_actor_snapshot(modules)

    replacement["backbone"]["weight"] = replacement["backbone"]["weight"] + 3.0
    replacement["head"]["bias"] = replacement["head"]["bias"] - 1.0
    replacement["log_std"] = replacement["log_std"] + 0.7

    with use_actor_snapshot(modules, replacement, device=torch.device("cpu")):
        inside = capture_actor_snapshot(modules)
        assert torch.equal(inside["backbone"]["weight"], replacement["backbone"]["weight"])
        assert torch.equal(inside["head"]["bias"], replacement["head"]["bias"])
        assert np.allclose(inside["log_std"], replacement["log_std"])

    after = capture_actor_snapshot(modules)
    assert torch.equal(after["backbone"]["weight"], original["backbone"]["weight"])
    assert torch.equal(after["head"]["bias"], original["head"]["bias"])
    assert np.allclose(after["log_std"], original["log_std"])


def test_sac_actor_eval_policy_and_snapshot_roundtrip():
    modules = _make_actor_modules()
    policy = SacActorEvalPolicy(
        modules.actor_backbone,
        modules.actor_head,
        modules.obs_scaler,
        act_dim=2,
        device=torch.device("cpu"),
    )
    action = policy(np.asarray([0.5, 0.0, -0.2], dtype=np.float32))
    assert isinstance(action, np.ndarray)
    assert action.shape == (2,)

    snapshot = capture_sac_actor_snapshot(modules)
    with torch.no_grad():
        modules.actor_backbone.weight.mul_(0.0)
        modules.actor_head.bias.add_(5.0)

    restore_sac_actor_snapshot(modules, snapshot)
    assert torch.equal(modules.actor_backbone.state_dict()["weight"], snapshot["backbone"]["weight"])
    assert torch.equal(modules.actor_head.state_dict()["bias"], snapshot["head"]["bias"])


def test_actor_eval_policy_atari_hwc4_discrete():
    backbone = nn.Sequential(
        nn.Conv2d(4, 8, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    head = nn.Linear(8, 6)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.arange(6, dtype=torch.float32))

    policy = ActorEvalPolicy(
        backbone,
        head,
        nn.Identity(),
        device=torch.device("cpu"),
        obs_contract=ObservationContract(mode="pixels", raw_shape=(84, 84, 4), model_channels=4, image_size=84),
        is_discrete=True,
    )
    atari_state_hwc = np.random.randint(0, 256, size=(84, 84, 4), dtype=np.uint8)
    action = policy(atari_state_hwc)
    assert np.asarray(action).shape == ()
    assert int(action) == 5
