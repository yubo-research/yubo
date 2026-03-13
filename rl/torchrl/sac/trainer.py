from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import tensordict.nn as td_nn
import torch
import torchrl.collectors as tr_collectors
import torchrl.envs as tr_envs

import common.video as video
from common.seed_all import seed_all
from rl.checkpointing import load_checkpoint
from rl.core import envs, runtime
from rl.core.actor_state import load, restore_rng_state_payload, snap
from rl.core.update_chunks import run_chunked_updates
from rl.env_provider import get_env_conf_fn
from rl.eval_noise import normalize_eval_noise_mode
from rl.registry import register_algo
from rl.torchrl.offpolicy import actor_eval, trainer_utils

from . import loop as torchrl_sac_loop
from .config import _SAC_RUNTIME_CAPABILITIES, SACConfig, TrainResult
from .setup import (
    _EnvSetup,
    _make_collect_env_sac,
    _Modules,
    _scale_action_to_env,
    _ScaleActionToEnv,
    _TrainingSetup,
    _TrainState,
    _unscale_action_from_env,
    build_env_setup,
    build_modules,
    build_training,
    sac_update_shared,
)


def _checkpoint_payload(modules: _Modules, training: _TrainingSetup, state: _TrainState, *, step: int) -> dict:
    actor_snapshot = snap(modules.actor_backbone, modules.actor_head, log_std=None, state_to_cpu=False)
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


def _resume_if_requested(
    config: SACConfig,
    modules: _Modules,
    training: _TrainingSetup,
    *,
    device: torch.device,
) -> _TrainState:
    state = _TrainState()
    if not config.resume_from:
        return state
    loaded = load_checkpoint(Path(config.resume_from), device=device)
    if "actor_backbone" in loaded and "actor_head" in loaded:
        load(
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


def _build_eval_policy(modules: _Modules, env_setup: _EnvSetup, device: torch.device):
    return actor_eval.OffPolicyActorEvalPolicy(
        modules.actor_backbone,
        modules.actor_head,
        modules.obs_scaler,
        act_dim=env_setup.act_dim,
        device=device,
        from_pixels=str(getattr(env_setup.env_conf, "obs_mode", "vector")) in {"image", "mixed", "pixels", "pixels+state"},
    )


def _build_sac_collector(
    config: SACConfig, env_setup: _EnvSetup, modules: _Modules, *, rt, total_frames: int
) -> tr_collectors.Collector | tr_collectors.MultiSyncCollector | tr_collectors.MultiAsyncCollector:
    frames_per_batch = int(config.frames_per_batch)
    num_envs = int(config.runtime_num_envs())
    scale_action = td_nn.TensorDictModule(
        _ScaleActionToEnv(env_setup.action_low, env_setup.action_high),
        in_keys=["action"],
        out_keys=["action"],
    ).to(rt.device)
    collector_policy = td_nn.TensorDictSequential(modules.actor, scale_action)
    if rt.collector_backend == "single":
        if num_envs == 1:
            vec_env = _make_collect_env_sac(env_setup.env_conf, env_setup, env_index=0)
        else:
            env_makers = [lambda i=i: _make_collect_env_sac(env_setup.env_conf, env_setup, env_index=i) for i in range(num_envs)]
            vec_env = (
                tr_envs.ParallelEnv(num_envs, env_makers, serial_for_single=True)
                if rt.single_env_backend == "parallel"
                else tr_envs.SerialEnv(num_envs, env_makers, serial_for_single=True)
            )
        return tr_collectors.Collector(
            vec_env,
            collector_policy,
            frames_per_batch=frames_per_batch * num_envs,
            total_frames=total_frames,
            init_random_frames=int(config.learning_starts),
            reset_at_each_iter=False,
            **runtime.collector_device_kwargs(rt.device),
        )
    num_workers = int(rt.collector_workers or num_envs)
    create_env_fns = [lambda i=i: _make_collect_env_sac(env_setup.env_conf, env_setup, env_index=i) for i in range(num_workers)]
    frames_per_batch_per_worker = max(1, frames_per_batch)
    collector_cls = tr_collectors.MultiAsyncCollector if rt.collector_backend == "multi_async" else tr_collectors.MultiSyncCollector
    return collector_cls(
        create_env_fn=create_env_fns,
        policy=collector_policy,
        frames_per_batch=[int(frames_per_batch_per_worker)] * num_workers,
        total_frames=total_frames,
        init_random_frames=int(config.learning_starts),
        reset_at_each_iter=False,
        env_device=torch.device("cpu"),
        policy_device=rt.device,
        storing_device=torch.device("cpu"),
    )


def _update_step(
    config: SACConfig,
    modules: _Modules,
    training: _TrainingSetup,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    batch = training.replay.sample(batch_size).to(device)
    obs = batch["observation"]
    act = batch["action"]
    nxt = batch["next", "observation"]
    rew = batch["next", "reward"]
    done = batch["next", "done"]
    if rew.ndim > 1:
        rew = rew.squeeze(-1)
    if done.ndim > 1:
        done = done.squeeze(-1)
    actor_loss, critic_loss, alpha_loss = sac_update_shared(
        config,
        modules,
        training,
        obs=obs,
        act=act,
        rew=rew.to(dtype=torch.float32),
        nxt=nxt,
        done=done.to(dtype=torch.float32),
    )
    return {
        "loss_actor": float(actor_loss),
        "loss_critic": float(critic_loss),
        "loss_alpha": float(alpha_loss),
        "alpha": float(modules.log_alpha.exp().detach().cpu()),
        "entropy": float("nan"),
    }


def _flatten_batch_to_transitions(batch):
    return trainer_utils.flatten_batch_to_transitions(batch)


def _normalize_actions_for_replay(flat, *, action_low: np.ndarray, action_high: np.ndarray):
    return trainer_utils.normalize_actions_for_replay(flat, action_low=action_low, action_high=action_high)


def _run_sac_eval_log_checkpoint(
    config,
    env,
    modules,
    training,
    state,
    step,
    rt,
    start_time,
    latest_losses,
    total_updates,
    best,
):
    get_env_conf = get_env_conf_fn()

    def _eval_heldout(cfg, env_setup, local_modules, local_state, *, device, heldout_i_noise=99999):
        return torchrl_sac_loop.heldout(
            cfg,
            env_setup,
            local_modules,
            local_state,
            device=device,
            heldout_i_noise=heldout_i_noise,
            capture_actor_state=actor_eval.capture_actor_snapshot,
            restore_actor_state=actor_eval.restore_actor_snapshot,
            eval_policy_factory=lambda a, e, d: _build_eval_policy(a, e, d),
            get_env_conf=get_env_conf,
            best=best,
        )

    torchrl_sac_loop.evaluate_if_due(
        config,
        env,
        modules,
        training,
        state,
        step=step,
        device=rt.device,
        start_time=start_time,
        latest_losses=latest_losses,
        total_updates=total_updates,
        evaluate_actor=lambda cfg, env_setup, local_modules, *, device, eval_seed: float(
            __import__("rl.core.episode_rollout", fromlist=["denoise"])
            .denoise(
                env_setup.env_conf,
                _build_eval_policy(local_modules, env_setup, device),
                num_denoise=cfg.num_denoise,
                i_noise=int(eval_seed),
            )[0]
            .rreturn
        ),
        capture_actor_state=actor_eval.capture_actor_snapshot,
        evaluate_heldout=_eval_heldout,
    )
    torchrl_sac_loop.log_if_due(
        config,
        state,
        step=step,
        start_time=start_time,
        latest_losses=latest_losses,
        total_updates=total_updates,
    )
    torchrl_sac_loop.checkpoint_if_due(
        config,
        modules,
        training,
        state,
        step=step,
        build_checkpoint_payload=_checkpoint_payload,
    )


def _process_sac_batch(batch, config, modules, training, rt, env_setup, latest_losses, total_updates):
    def _update_step_with_chunking(device: torch.device, batch_size: int) -> dict[str, float]:
        nonlocal latest_losses, total_updates

        def _run_one_update() -> None:
            nonlocal latest_losses, total_updates
            if training.replay.write_count >= int(config.learning_starts):
                latest_losses = _update_step(config, modules, training, device=device, batch_size=batch_size)
                total_updates += 1

        run_chunked_updates(1, int(config.learner_update_chunk_size), _run_one_update)
        return latest_losses

    latest_losses, total_updates, n_frames = trainer_utils.process_offpolicy_batch(
        batch,
        config=config,
        training=training,
        runtime_device=rt.device,
        env_setup=env_setup,
        latest_losses=latest_losses,
        total_updates=total_updates,
        update_step_fn=_update_step_with_chunking,
    )
    return (latest_losses, total_updates, int(n_frames))


def train_sac(config: SACConfig) -> TrainResult:
    with runtime.temporary_distribution_validate_args(False):
        from rl.core.episode_rollout import best

        if config.eval_noise_mode is not None:
            normalize_eval_noise_mode(config.eval_noise_mode)
        resolved = envs.seeds(
            seed=int(config.seed),
            problem_seed=config.problem_seed,
            noise_seed_0=config.noise_seed_0,
        )
        config.problem_seed = int(resolved.problem_seed)
        config.noise_seed_0 = int(resolved.noise_seed_0)
        seed_all(envs.global_seed_for_run(int(resolved.problem_seed)))
        env = build_env_setup(config)
        rt = config.resolve_runtime(capabilities=_SAC_RUNTIME_CAPABILITIES)
        modules = build_modules(config, env, device=rt.device)
        training = build_training(config, modules)
        state = _resume_if_requested(config, modules, training, device=rt.device)
        from rl import logger as rl_logger

        rl_logger.log_run_header("sac", config, env, training, rt)
        total_frames = int(config.total_timesteps) - state.start_step
        if total_frames <= 0:
            total_frames = 1
        collector = _build_sac_collector(config, env, modules, rt=rt, total_frames=total_frames)
        collector.set_seed(int(env.problem_seed))
        if hasattr(collector, "shutdown"):
            pass
        start_time = time.time()
        latest_losses = {
            "loss_actor": float("nan"),
            "loss_critic": float("nan"),
            "loss_alpha": float("nan"),
        }
        total_updates = 0
        step = state.start_step
        for batch in collector:
            latest_losses, total_updates, n_frames = _process_sac_batch(batch, config, modules, training, rt, env, latest_losses, total_updates)
            step += n_frames
            if step >= int(config.total_timesteps):
                step = int(config.total_timesteps)
                _run_sac_eval_log_checkpoint(
                    config,
                    env,
                    modules,
                    training,
                    state,
                    step,
                    rt,
                    start_time,
                    latest_losses,
                    total_updates,
                    best,
                )
                break
            _run_sac_eval_log_checkpoint(
                config,
                env,
                modules,
                training,
                state,
                step,
                rt,
                start_time,
                latest_losses,
                total_updates,
                best,
            )
        try:
            collector.shutdown()
        except Exception:
            pass
        total_time = time.time() - start_time
        best_return = float("nan") if state.best_return == -float("inf") else float(state.best_return)
        rl_logger.log_run_footer(
            best_return,
            int(config.total_timesteps),
            total_time,
            algo_name="sac",
            step_label="steps",
        )
        torchrl_sac_loop.save_final_checkpoint_if_enabled(
            config,
            modules,
            training,
            state,
            build_checkpoint_payload=_checkpoint_payload,
        )
        if config.video_enable:
            ctx = video.RLVideoContext(
                env=lambda ps, ns: get_env_conf_fn()(config.env_tag, problem_seed=ps, noise_seed_0=ns),
                make_eval_policy=lambda m, d: _build_eval_policy(m, env, d),
                capture_actor_state=actor_eval.capture_actor_snapshot,
                with_actor_state=actor_eval.use_actor_snapshot,
            )
            video.render_policy_videos_rl(config, env, modules, training, state, ctx, device=rt.device)
        return TrainResult(
            best_return=best_return,
            last_eval_return=float(state.last_eval_return),
            last_heldout_return=state.last_heldout_return,
            num_steps=int(config.total_timesteps),
        )


def register():
    register_algo("sac", SACConfig, train_sac)


__all__ = [
    "SACConfig",
    "TrainResult",
    "_scale_action_to_env",
    "_unscale_action_from_env",
    "register",
    "train_sac",
]
