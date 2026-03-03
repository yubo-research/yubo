from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from tensordict import TensorDict

from rl.core.continuous_actions import unscale_action_tensor_from_env


def flatten_batch_to_transitions(batch: TensorDict) -> TensorDict:
    flat = batch.reshape(-1)
    if "next" in flat.keys():
        next_td = flat["next"]
        if "done" not in next_td.keys():
            term = next_td.get("terminated", torch.zeros(*next_td.batch_size, 1, dtype=torch.bool, device=next_td.device))
            trunc = next_td.get("truncated", torch.zeros(*next_td.batch_size, 1, dtype=torch.bool, device=next_td.device))
            next_td = next_td.set("done", term | trunc)
        for key in ("reward", "done", "terminated"):
            if key in next_td.keys() and next_td[key].ndim == 1:
                next_td = next_td.set(key, next_td[key].unsqueeze(-1))
        flat = flat.set("next", next_td)
    return flat


def normalize_actions_for_replay(flat: TensorDict, *, action_low: np.ndarray, action_high: np.ndarray) -> TensorDict:
    if "action" not in flat.keys():
        return flat
    action = flat["action"]
    low = torch.as_tensor(action_low, dtype=action.dtype, device=action.device)
    high = torch.as_tensor(action_high, dtype=action.dtype, device=action.device)
    action_norm = unscale_action_tensor_from_env(action, low, high, clip=True)
    return flat.set("action", action_norm)


def process_offpolicy_batch(
    batch: TensorDict,
    *,
    config: Any,
    training: Any,
    runtime_device: torch.device,
    env_setup: Any,
    latest_losses: dict[str, float],
    total_updates: int,
    update_step_fn: Callable[[torch.device, int], dict[str, float]],
) -> tuple[dict[str, float], int, int]:
    flat = flatten_batch_to_transitions(batch)
    flat = normalize_actions_for_replay(flat, action_low=env_setup.action_low, action_high=env_setup.action_high)
    n_frames = int(flat.shape[0]) if flat.ndim > 0 else 1
    for i in range(n_frames):
        training.replay.add(flat[i].clone())
    n_update_cycles = max(0, n_frames // int(config.update_every))
    for _ in range(n_update_cycles * int(config.updates_per_step)):
        if training.replay.write_count >= int(config.learning_starts):
            latest_losses = update_step_fn(runtime_device, int(config.batch_size))
            total_updates += 1
    return (latest_losses, total_updates, n_frames)
