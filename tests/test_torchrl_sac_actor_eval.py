import numpy as np
import torch
import torch.nn as nn

from rl.backends.torchrl.sac.actor_eval import (
    SacActorEvalPolicy,
    capture_sac_actor_snapshot,
    use_sac_actor_snapshot,
)


def test_use_sac_actor_snapshot_restores_on_exit():
    backbone = nn.Linear(2, 3)
    head = nn.Linear(3, 1)
    modules = type("M", (), {"actor_backbone": backbone, "actor_head": head})()

    with torch.inference_mode():
        backbone.weight.fill_(1.0)
        head.weight.fill_(2.0)
    snapshot = capture_sac_actor_snapshot(modules)

    with torch.inference_mode():
        backbone.weight.fill_(99.0)
        head.weight.fill_(88.0)

    with use_sac_actor_snapshot(modules, snapshot, device=torch.device("cpu")):
        torch.testing.assert_close(backbone.weight, torch.ones_like(backbone.weight))
        torch.testing.assert_close(head.weight, torch.full_like(head.weight, 2.0))

    torch.testing.assert_close(backbone.weight, torch.full_like(backbone.weight, 99.0))
    torch.testing.assert_close(head.weight, torch.full_like(head.weight, 88.0))


def test_sac_actor_eval_policy_handles_pixel_input():
    backbone = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    head = nn.Linear(4, 4)
    policy = SacActorEvalPolicy(
        backbone,
        head,
        nn.Identity(),
        act_dim=2,
        device=torch.device("cpu"),
        from_pixels=True,
    )
    action = policy(np.zeros((84, 84, 3), dtype=np.uint8))
    assert isinstance(action, np.ndarray)
    assert action.shape == (2,)
