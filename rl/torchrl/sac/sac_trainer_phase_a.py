from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import torch

from rl.checkpointing import load_checkpoint

from . import actor_eval
from .config import SACConfig
from .setup import _EnvSetup, _Modules, _TrainingSetup, _TrainState


def checkpoint_payload(modules: _Modules, training: _TrainingSetup, state: _TrainState, *, step: int) -> dict:
    capture_backbone_head_snapshot = getattr(importlib.import_module("rl.core.actor_state"), "capture_backbone_head_snapshot")
    actor_snapshot = capture_backbone_head_snapshot(modules.actor_backbone, modules.actor_head, log_std=None, state_to_cpu=False)
    return {
        "step": int(step),
        "actor_backbone": actor_snapshot["backbone"],
        "actor_head": actor_snapshot["head"],
        "obs_scaler": modules.obs_scaler.state_dict(),
        "q1": modules.q1.state_dict(),
        "q2": modules.q2.state_dict(),
        "q1_target": modules.q1_target.state_dict(),
        "q2_target": modules.q2_target.state_dict(),
        "log_alpha": modules.log_alpha.detach().cpu(),
        "replay_state": training.replay.state_dict(),
        "actor_optimizer": training.actor_optimizer.state_dict(),
        "critic_optimizer": training.critic_optimizer.state_dict(),
        "alpha_optimizer": training.alpha_optimizer.state_dict(),
        "best_return": float(state.best_return),
        "best_actor_state": state.best_actor_state,
        "last_eval_return": float(state.last_eval_return),
        "last_heldout_return": state.last_heldout_return,
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _load_state_if_present(loaded: dict, key: str, module: torch.nn.Module) -> None:
    state = loaded.get(key)
    if state is not None:
        module.load_state_dict(state)


def _copy_if_present(loaded: dict, key: str, target: torch.Tensor, *, device: torch.device) -> None:
    value = loaded.get(key)
    if value is not None:
        target.copy_(value.to(device=device, dtype=target.dtype))


def resume_if_requested(
    config: SACConfig,
    modules: _Modules,
    training: _TrainingSetup,
    *,
    device: torch.device,
) -> _TrainState:
    state = _TrainState()
    if not config.checkpoint.resume_from:
        return state
    _actor = importlib.import_module("rl.core.actor_state")
    restore_backbone_head_snapshot = getattr(_actor, "restore_backbone_head_snapshot")
    restore_rng_state_payload = getattr(_actor, "restore_rng_state_payload")
    loaded = load_checkpoint(Path(config.checkpoint.resume_from), device=device)
    if "actor_backbone" in loaded and "actor_head" in loaded:
        restore_backbone_head_snapshot(
            modules.actor_backbone,
            modules.actor_head,
            {"backbone": loaded["actor_backbone"], "head": loaded["actor_head"]},
            log_std=None,
            device=device,
        )
    _load_state_if_present(loaded, "obs_scaler", modules.obs_scaler)
    _load_state_if_present(loaded, "q1", modules.q1)
    _load_state_if_present(loaded, "q2", modules.q2)
    _load_state_if_present(loaded, "q1_target", modules.q1_target)
    _load_state_if_present(loaded, "q2_target", modules.q2_target)
    _copy_if_present(loaded, "log_alpha", modules.log_alpha.data, device=device)
    replay_state = loaded.get("replay_state")
    if replay_state is not None:
        training.replay.load_state_dict(replay_state)
    training.actor_optimizer.load_state_dict(loaded["actor_optimizer"])
    training.critic_optimizer.load_state_dict(loaded["critic_optimizer"])
    training.alpha_optimizer.load_state_dict(loaded["alpha_optimizer"])
    state.start_step = int(loaded.get("step", 0))
    state.best_return = float(loaded.get("best_return", state.best_return))
    state.best_actor_state = loaded.get("best_actor_state")
    state.last_eval_return = float(loaded.get("last_eval_return", state.last_eval_return))
    state.last_heldout_return = loaded.get("last_heldout_return", state.last_heldout_return)
    restore_rng_state_payload(loaded)
    return state


def build_eval_policy(modules: _Modules, env_setup: _EnvSetup, device: torch.device):
    return actor_eval.SacActorEvalPolicy(
        modules.actor_backbone,
        modules.actor_head,
        modules.obs_scaler,
        act_dim=env_setup.act_dim,
        device=device,
        from_pixels=bool(getattr(env_setup.env_conf, "from_pixels", False)),
    )


def evaluate_actor(
    config: SACConfig,
    env: _EnvSetup,
    modules: _Modules,
    *,
    device: torch.device,
    eval_seed: int,
) -> float:
    collect_denoised_trajectory = getattr(
        importlib.import_module("rl.core.episode_rollout"),
        "collect_denoised_trajectory",
    )

    eval_env = env.env_conf
    eval_policy = build_eval_policy(modules, env, device)
    traj, _ = collect_denoised_trajectory(
        eval_env,
        eval_policy,
        num_denoise=config.eval.num_denoise,
        i_noise=int(eval_seed),
    )
    return float(traj.rreturn)
