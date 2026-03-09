from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

from rl.torchrl.offpolicy import actor_eval, trainer_utils


def test_offpolicy_actor_eval_policy_vector_and_pixels():
    vector_backbone = nn.Linear(3, 4)
    vector_head = nn.Linear(4, 4)
    vector_policy = actor_eval.OffPolicyActorEvalPolicy(
        vector_backbone,
        vector_head,
        nn.Identity(),
        act_dim=2,
        device=torch.device("cpu"),
        from_pixels=False,
    )
    out_vec = vector_policy(np.asarray([0.1, 0.2, -0.3], dtype=np.float32))
    assert out_vec.shape == (2,)

    pixel_backbone = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    pixel_head = nn.Linear(4, 6)
    pixel_policy = actor_eval.OffPolicyActorEvalPolicy(
        pixel_backbone,
        pixel_head,
        nn.Identity(),
        act_dim=3,
        device=torch.device("cpu"),
        from_pixels=True,
        channels=3,
        image_size=84,
    )
    out_px = pixel_policy(np.random.randint(0, 255, size=(84, 84, 3), dtype=np.uint8))
    assert out_px.shape == (3,)


def test_offpolicy_actor_snapshot_helpers_roundtrip():
    modules = SimpleNamespace(
        actor_backbone=nn.Linear(3, 4),
        actor_head=nn.Linear(4, 2),
    )
    snap = actor_eval.capture_actor_snapshot(modules)

    with torch.no_grad():
        modules.actor_backbone.weight.add_(5.0)
        modules.actor_head.bias.sub_(3.0)

    actor_eval.restore_actor_snapshot(modules, snap)
    assert torch.equal(modules.actor_backbone.state_dict()["weight"], snap["backbone"]["weight"])
    assert torch.equal(modules.actor_head.state_dict()["bias"], snap["head"]["bias"])

    changed = actor_eval.capture_actor_snapshot(modules)
    changed["backbone"]["weight"] = changed["backbone"]["weight"] + 1.0
    with actor_eval.use_actor_snapshot(modules, changed, device=torch.device("cpu")):
        inside = actor_eval.capture_actor_snapshot(modules)
        assert torch.equal(inside["backbone"]["weight"], changed["backbone"]["weight"])


def test_trainer_utils_flatten_normalize_and_process():
    batch = TensorDict(
        {
            "obs": torch.zeros(2, 3, dtype=torch.float32),
            "action": torch.tensor([[0.0, 1.0], [0.5, -0.5]], dtype=torch.float32),
            "next": TensorDict(
                {
                    "reward": torch.tensor([1.0, 2.0], dtype=torch.float32),
                    "terminated": torch.tensor([True, False]),
                    "truncated": torch.tensor([False, True]),
                },
                batch_size=[2],
            ),
        },
        batch_size=[2],
    )

    flat = trainer_utils.flatten_batch_to_transitions(batch)
    assert flat["next"]["done"].shape[-1] == 1
    assert flat["next"]["reward"].shape[-1] == 1

    norm = trainer_utils.normalize_actions_for_replay(
        flat,
        action_low=np.asarray([-2.0, -1.0], dtype=np.float32),
        action_high=np.asarray([2.0, 1.0], dtype=np.float32),
    )
    assert "action" in norm.keys()

    no_action = TensorDict({"obs": torch.zeros(2, 3)}, batch_size=[2])
    assert (
        trainer_utils.normalize_actions_for_replay(
            no_action,
            action_low=np.asarray([-1.0], dtype=np.float32),
            action_high=np.asarray([1.0], dtype=np.float32),
        )
        is no_action
    )

    class _Replay:
        def __init__(self):
            self.items = []
            self.write_count = 10

        def add(self, item):
            self.items.append(item)

    replay = _Replay()
    training = SimpleNamespace(replay=replay)
    latest = {"loss_actor": float("nan")}
    calls = {"n": 0}

    def _update_step(_device, _batch_size):
        calls["n"] += 1
        return {"loss_actor": float(calls["n"])}

    out_losses, total_updates, n_frames = trainer_utils.process_offpolicy_batch(
        batch,
        config=SimpleNamespace(update_every=1, updates_per_step=2, learning_starts=0, batch_size=4),
        training=training,
        runtime_device=torch.device("cpu"),
        env_setup=SimpleNamespace(
            action_low=np.asarray([-2.0, -1.0], dtype=np.float32),
            action_high=np.asarray([2.0, 1.0], dtype=np.float32),
        ),
        latest_losses=latest,
        total_updates=3,
        update_step_fn=_update_step,
    )

    assert n_frames == 2
    assert len(replay.items) == 2
    assert calls["n"] == 4
    assert total_updates == 7
    assert out_losses["loss_actor"] == 4.0
