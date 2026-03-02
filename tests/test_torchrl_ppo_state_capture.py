from types import SimpleNamespace

import torch
import torch.nn as nn

from rl.torchrl.ppo.actor_eval import (
    capture_actor_snapshot,
    restore_actor_snapshot,
)


def test_capture_actor_state_is_immutable_snapshot():
    actor_backbone = nn.Linear(3, 4)
    actor_head = nn.Linear(4, 2)
    log_std = nn.Parameter(torch.zeros(2))
    modules = SimpleNamespace(actor_backbone=actor_backbone, actor_head=actor_head, log_std=log_std)

    snapshot = capture_actor_snapshot(modules)

    with torch.no_grad():
        actor_backbone.weight.add_(1.0)
        actor_head.bias.sub_(2.0)
        log_std.add_(0.5)

    assert not torch.equal(snapshot["backbone"]["weight"], actor_backbone.state_dict()["weight"])
    assert not torch.equal(snapshot["head"]["bias"], actor_head.state_dict()["bias"])
    assert float(snapshot["log_std"][0]) != float(log_std[0])

    restore_actor_snapshot(modules, snapshot, device=torch.device("cpu"))
    assert torch.equal(actor_backbone.state_dict()["weight"], snapshot["backbone"]["weight"])
    assert torch.equal(actor_head.state_dict()["bias"], snapshot["head"]["bias"])
    assert torch.allclose(log_std, torch.as_tensor(snapshot["log_std"]))
