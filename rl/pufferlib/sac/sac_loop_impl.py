from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch


if TYPE_CHECKING:
    from .config import SACConfig

_sac = "rl.pufferlib.sac"


def _eval_utils():
    return importlib.import_module(f"{_sac}.eval_utils")


def _env_utils():
    return importlib.import_module(f"{_sac}.env_utils")


def _model_utils():
    return importlib.import_module(f"{_sac}.model_utils")


def _state_payload(modules, optimizers, replay, state: Any) -> dict[str, Any]:
    actor_state_mod = importlib.import_module("rl.core.actor_state")
    capture_backbone_head_snapshot = getattr(actor_state_mod, "capture_backbone_head_snapshot")
    actor_snapshot = capture_backbone_head_snapshot(modules.actor_backbone, modules.actor_head, log_std=None, state_to_cpu=False)
    return {
        "step": int(state.global_step),
        "total_updates": int(state.total_updates),
        "actor_backbone": actor_snapshot["backbone"],
        "actor_head": actor_snapshot["head"],
        "q1": modules.q1.state_dict(),
        "q2": modules.q2.state_dict(),
        "q1_target": modules.q1_target.state_dict(),
        "q2_target": modules.q2_target.state_dict(),
        "obs_scaler": modules.obs_scaler.state_dict(),
        "log_alpha": modules.log_alpha.detach().cpu(),
        "actor_optimizer": optimizers.actor_optimizer.state_dict(),
        "critic_optimizer": optimizers.critic_optimizer.state_dict(),
        "alpha_optimizer": optimizers.alpha_optimizer.state_dict(),
        "best_return": float(state.best_return),
        "best_actor_state": state.best_actor_state,
        "last_eval_return": float(state.last_eval_return),
        "last_heldout_return": state.last_heldout_return,
        "replay_state": replay.state_dict(),
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_if_requested(config, modules, optimizers, replay, state: Any, *, device) -> None:
    if not config.resume_from:
        return
    actor_state_mod = importlib.import_module("rl.core.actor_state")
    load_checkpoint = importlib.import_module("rl.checkpointing").load_checkpoint
    restore_backbone_head_snapshot = getattr(actor_state_mod, "restore_backbone_head_snapshot")
    restore_rng_state_payload = getattr(actor_state_mod, "restore_rng_state_payload")
    loaded = load_checkpoint(Path(config.resume_from), device=device)
    restore_backbone_head_snapshot(
        modules.actor_backbone,
        modules.actor_head,
        {"backbone": loaded["actor_backbone"], "head": loaded["actor_head"]},
        log_std=None,
        device=device,
    )
    modules.q1.load_state_dict(loaded["q1"])
    modules.q2.load_state_dict(loaded["q2"])
    modules.q1_target.load_state_dict(loaded["q1_target"])
    modules.q2_target.load_state_dict(loaded["q2_target"])
    if loaded.get("obs_scaler") is not None:
        modules.obs_scaler.load_state_dict(loaded["obs_scaler"])
    if loaded.get("log_alpha") is not None:
        modules.log_alpha.data.copy_(loaded["log_alpha"].to(device=device, dtype=modules.log_alpha.dtype))
    optimizers.actor_optimizer.load_state_dict(loaded["actor_optimizer"])
    optimizers.critic_optimizer.load_state_dict(loaded["critic_optimizer"])
    optimizers.alpha_optimizer.load_state_dict(loaded["alpha_optimizer"])
    if loaded.get("replay_state") is not None:
        replay.load_state_dict(loaded["replay_state"])
    state.global_step = int(loaded.get("step", state.global_step))
    state.total_updates = int(loaded.get("total_updates", state.total_updates))
    state.best_return = float(loaded.get("best_return", state.best_return))
    state.best_actor_state = loaded.get("best_actor_state")
    state.last_eval_return = float(loaded.get("last_eval_return", state.last_eval_return))
    state.last_heldout_return = loaded.get("last_heldout_return", state.last_heldout_return)
    restore_rng_state_payload(loaded)


def _checkpoint_if_due(config, checkpoint_manager: Any, modules, optimizers, replay, state: Any) -> None:
    eu = _eval_utils()
    due_mark = eu.due_mark
    offpolicy_engine_utils = importlib.import_module("rl.pufferlib.offpolicy.engine_utils")
    state.ckpt_mark = offpolicy_engine_utils.checkpoint_mark_if_due(
        global_step=int(state.global_step),
        checkpoint_interval_steps=config.checkpoint_interval_steps,
        previous_mark=int(state.ckpt_mark),
        due_mark_fn=due_mark,
        save_fn=lambda: checkpoint_manager.save_both(
            _state_payload(modules, optimizers, replay, state),
            iteration=int(state.global_step),
        ),
    )


def _random_actions(num_envs: int, act_dim: int) -> np.ndarray:
    return np.random.uniform(low=-1.0, high=1.0, size=(int(num_envs), int(act_dim))).astype(np.float32)


def _train_loop(
    config: "SACConfig",
    env_setup,
    modules,
    optimizers,
    replay: Any,
    state: Any,
    obs_spec,
    obs_batch: np.ndarray,
    envs,
    *,
    device: torch.device,
    metrics_path: Path,
    checkpoint_manager: Any,
) -> None:
    envu = _env_utils()
    prepare_obs_np = envu.prepare_obs_np
    to_env_action = envu.to_env_action
    ev = _eval_utils()
    append_eval_metric = ev.append_eval_metric
    log_if_due = ev.log_if_due
    maybe_eval = ev.maybe_eval
    sac_update = _model_utils().sac_update
    run_chunked_updates = importlib.import_module("rl.core.update_chunks").run_chunked_updates
    total_steps = int(config.total_timesteps)
    num_envs = int(obs_batch.shape[0])
    while state.global_step < total_steps:
        if state.global_step < int(config.learning_starts):
            action_norm = _random_actions(num_envs, int(env_setup.act_dim))
        else:
            obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_t, _ = modules.actor.sample(obs_t, deterministic=False)
            action_norm = np.asarray(action_t.detach().cpu().numpy(), dtype=np.float32)
        action_env = to_env_action(action_norm, low=env_setup.action_low, high=env_setup.action_high)
        nxt_obs_np, reward_np, terminated_np, truncated_np, _ = envs.step(action_env)
        done_np = np.logical_or(terminated_np, truncated_np)
        nxt_obs = prepare_obs_np(nxt_obs_np, obs_spec=obs_spec)
        replay.add_batch(obs_batch, action_norm, reward_np, nxt_obs, done_np)
        obs_batch = nxt_obs
        state.global_step = int(min(total_steps, state.global_step + num_envs))
        should_update = state.global_step >= int(config.learning_starts) and state.global_step % int(max(1, config.update_every)) < num_envs
        if should_update and replay.size >= int(config.batch_size):

            def _run_one_update() -> None:
                actor_loss, critic_loss, alpha_loss = sac_update(config, modules, optimizers, replay, device=device)
                state.total_updates += 1
                state.last_loss_actor = float(actor_loss)
                state.last_loss_critic = float(critic_loss)
                state.last_loss_alpha = float(alpha_loss)

            run_chunked_updates(
                int(max(1, config.updates_per_step)),
                int(config.learner_update_chunk_size),
                _run_one_update,
            )
        prev_eval_mark = int(state.eval_mark)
        maybe_eval(config, env_setup, modules, obs_spec, state, device=device)
        if int(state.eval_mark) != prev_eval_mark:
            append_eval_metric(metrics_path, state, step=int(state.global_step))
        log_if_due(
            config,
            state,
            step=int(state.global_step),
            frames_per_batch=int(max(1, num_envs)),
        )
        _checkpoint_if_due(config, checkpoint_manager, modules, optimizers, replay, state)
