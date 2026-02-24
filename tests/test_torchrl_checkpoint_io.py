from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim

from rl.torchrl.ppo.checkpoint_io import (
    build_checkpoint_payload,
    save_final_checkpoint,
    save_periodic_checkpoint,
)


class _CheckpointManagerStub:
    def __init__(self):
        self.calls = []

    def save_both(self, payload, iteration):
        self.calls.append((payload, int(iteration)))


def _make_training_setup():
    optimizer_module = nn.Linear(2, 2)
    optimizer = optim.Adam(optimizer_module.parameters(), lr=1e-3)
    return SimpleNamespace(
        frames_per_batch=64,
        optimizer=optimizer,
        checkpoint_manager=_CheckpointManagerStub(),
        num_iterations=7,
    )


def _make_modules():
    return SimpleNamespace(
        actor_backbone=nn.Linear(3, 4),
        actor_head=nn.Linear(4, 2),
        critic_backbone=nn.Linear(5, 6),
        critic_head=nn.Linear(6, 1),
        obs_scaler=nn.Identity(),
        log_std=nn.Parameter(torch.zeros(2)),
    )


def _make_train_state():
    return SimpleNamespace(
        best_return=12.3,
        best_actor_state={"marker": 1},
        last_eval_return=8.1,
        last_heldout_return=7.5,
    )


def test_build_checkpoint_payload_fields():
    training_setup = _make_training_setup()
    modules = _make_modules()
    train_state = _make_train_state()

    payload = build_checkpoint_payload(training_setup, modules, train_state, iteration=3)
    assert payload["iteration"] == 3
    assert payload["global_step"] == 192
    assert "actor_backbone" in payload
    assert "critic_head" in payload
    assert payload["best_return"] == 12.3
    assert payload["last_eval_return"] == 8.1
    assert payload["last_heldout_return"] == 7.5


def test_save_periodic_checkpoint_due_and_not_due():
    training_setup = _make_training_setup()
    modules = _make_modules()
    train_state = _make_train_state()
    config = SimpleNamespace(checkpoint_interval=5)

    save_periodic_checkpoint(config, training_setup, modules, train_state, iteration=4)
    assert training_setup.checkpoint_manager.calls == []

    save_periodic_checkpoint(config, training_setup, modules, train_state, iteration=5)
    assert len(training_setup.checkpoint_manager.calls) == 1
    _, saved_iteration = training_setup.checkpoint_manager.calls[0]
    assert saved_iteration == 5


def test_save_final_checkpoint_respects_interval():
    training_setup = _make_training_setup()
    modules = _make_modules()
    train_state = _make_train_state()

    disabled_config = SimpleNamespace(checkpoint_interval=None)
    save_final_checkpoint(disabled_config, training_setup, modules, train_state)
    assert training_setup.checkpoint_manager.calls == []

    enabled_config = SimpleNamespace(checkpoint_interval=2)
    save_final_checkpoint(enabled_config, training_setup, modules, train_state)
    assert len(training_setup.checkpoint_manager.calls) == 1
    _, saved_iteration = training_setup.checkpoint_manager.calls[0]
    assert saved_iteration == training_setup.num_iterations
