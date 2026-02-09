from types import SimpleNamespace

import torch
import torch.nn as nn

from rl.algos.torchrl_ppo import _capture_actor_state, _restore_actor_state


def test_capture_actor_state_is_immutable_snapshot():
    actor_backbone = nn.Linear(3, 4)
    actor_head = nn.Linear(4, 2)
    log_std = nn.Parameter(torch.zeros(2))
    modules = SimpleNamespace(actor_backbone=actor_backbone, actor_head=actor_head, log_std=log_std)

    snapshot = _capture_actor_state(modules)

    with torch.no_grad():
        actor_backbone.weight.add_(1.0)
        actor_head.bias.sub_(2.0)
        log_std.add_(0.5)

    assert not torch.equal(snapshot["backbone"]["weight"], actor_backbone.state_dict()["weight"])
    assert not torch.equal(snapshot["head"]["bias"], actor_head.state_dict()["bias"])
    assert float(snapshot["log_std"][0]) != float(log_std[0])

    _restore_actor_state(modules, snapshot, device=torch.device("cpu"))
    assert torch.equal(actor_backbone.state_dict()["weight"], snapshot["backbone"]["weight"])
    assert torch.equal(actor_head.state_dict()["bias"], snapshot["head"]["bias"])
    assert torch.allclose(log_std, torch.as_tensor(snapshot["log_std"]))
