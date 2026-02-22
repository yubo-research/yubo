"""SAC training coordinator. Delegates setup to setup.py and eval/checkpoint to loop.py."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torchrl.collectors as tr_collectors
import torchrl.envs as tr_envs
from tensordict import TensorDict
from torchrl.envs.libs.gym import set_gym_backend

from . import deps as sac_deps
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
)


def _checkpoint_payload(
    modules: _Modules,
    training: _TrainingSetup,
    state: _TrainState,
    *,
    step: int,
) -> dict:
    return {
        "step": int(step),
        "actor_backbone": modules.actor_backbone.state_dict(),
        "actor_head": modules.actor_head.state_dict(),
        "obs_scaler": modules.obs_scaler.state_dict(),
        "replay_state": training.replay.state_dict(),
        "loss_module": training.loss_module.state_dict(),
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
    loaded = sac_deps.load_checkpoint(Path(config.resume_from), device=device)
    modules.actor_backbone.load_state_dict(loaded["actor_backbone"])
    modules.actor_head.load_state_dict(loaded["actor_head"])
    if loaded.get("obs_scaler") is not None:
        modules.obs_scaler.load_state_dict(loaded["obs_scaler"])
    if loaded.get("replay_state") is not None:
        training.replay.load_state_dict(loaded["replay_state"])
    if loaded.get("loss_module") is not None:
        training.loss_module.load_state_dict(loaded["loss_module"])
    training.actor_optimizer.load_state_dict(loaded["actor_optimizer"])
    training.critic_optimizer.load_state_dict(loaded["critic_optimizer"])
    training.alpha_optimizer.load_state_dict(loaded["alpha_optimizer"])
    state.start_step = int(loaded.get("step", 0))
    state.best_return = float(loaded.get("best_return", state.best_return))
    state.best_actor_state = loaded.get("best_actor_state")
    state.last_eval_return = float(loaded.get("last_eval_return", state.last_eval_return))
    state.last_heldout_return = loaded.get("last_heldout_return", state.last_heldout_return)
    if "rng_torch" in loaded:
        torch.set_rng_state(loaded["rng_torch"])
    if "rng_numpy" in loaded:
        np.random.set_state(loaded["rng_numpy"])
    if torch.cuda.is_available() and loaded.get("rng_cuda") is not None:
        torch.cuda.set_rng_state_all(loaded["rng_cuda"])
    return state


def _evaluate_actor(
    config: SACConfig,
    env: _EnvSetup,
    modules: _Modules,
    *,
    device: torch.device,
    eval_seed: int,
) -> float:
    from optimizer.opt_trajectories import collect_denoised_trajectory

    eval_env = env.env_conf
    from_pixels = bool(getattr(eval_env, "from_pixels", False))
    eval_policy = sac_deps.torchrl_sac_actor_eval.SacActorEvalPolicy(
        modules.actor_backbone,
        modules.actor_head,
        modules.obs_scaler,
        act_dim=env.act_dim,
        device=device,
        from_pixels=from_pixels,
    )
    traj, _ = collect_denoised_trajectory(
        eval_env,
        eval_policy,
        num_denoise=config.num_denoise_eval,
        i_noise=int(eval_seed),
    )
    return float(traj.rreturn)


def _build_sac_collector(
    config: SACConfig,
    env_setup: _EnvSetup,
    modules: _Modules,
    *,
    runtime,
    total_frames: int,
) -> tr_collectors.Collector | tr_collectors.MultiSyncCollector | tr_collectors.MultiAsyncCollector:
    frames_per_batch = int(config.frames_per_batch)
    num_envs = int(config.runtime_num_envs())
    scale_action = sac_deps.td_nn.TensorDictModule(
        _ScaleActionToEnv(env_setup.action_low, env_setup.action_high),
        in_keys=["action"],
        out_keys=["action"],
    ).to(runtime.device)
    collector_policy = sac_deps.td_nn.TensorDictSequential(modules.actor, scale_action)

    if runtime.collector_backend == "single":
        if num_envs == 1:
            vec_env = _make_collect_env_sac(env_setup.env_conf, env_setup, env_index=0)
        else:
            env_makers = [lambda i=i: _make_collect_env_sac(env_setup.env_conf, env_setup, env_index=i) for i in range(num_envs)]
            vec_env = (
                tr_envs.ParallelEnv(num_envs, env_makers, serial_for_single=True)
                if runtime.single_env_backend == "parallel"
                else tr_envs.SerialEnv(num_envs, env_makers, serial_for_single=True)
            )
        return tr_collectors.Collector(
            vec_env,
            collector_policy,
            frames_per_batch=frames_per_batch * num_envs,
            total_frames=total_frames,
            init_random_frames=int(config.learning_starts),
            reset_at_each_iter=False,
            **sac_deps.torchrl_common.collector_device_kwargs(runtime.device),
        )

    num_workers = int(runtime.collector_workers or num_envs)
    create_env_fns = [(lambda i=i: _make_collect_env_sac(env_setup.env_conf, env_setup, env_index=i)) for i in range(num_workers)]
    frames_per_batch_per_worker = max(1, frames_per_batch)
    collector_cls = tr_collectors.MultiAsyncCollector if runtime.collector_backend == "multi_async" else tr_collectors.MultiSyncCollector
    return collector_cls(
        create_env_fn=create_env_fns,
        policy=collector_policy,
        frames_per_batch=[int(frames_per_batch_per_worker)] * num_workers,
        total_frames=total_frames,
        init_random_frames=int(config.learning_starts),
        reset_at_each_iter=False,
        env_device=torch.device("cpu"),
        policy_device=runtime.device,
        storing_device=torch.device("cpu"),
    )


def _flatten_batch_to_transitions(batch: TensorDict) -> TensorDict:
    """Flatten Collector batch (B, ...) to transitions for replay. Ensure next.done exists.
    Match our legacy format: next.reward and next.done as (B, 1) for replay compatibility."""
    flat = batch.reshape(-1)
    if "next" in flat.keys():
        next_td = flat["next"]
        if "done" not in next_td.keys():
            term = next_td.get(
                "terminated",
                torch.zeros(*next_td.batch_size, 1, dtype=torch.bool, device=next_td.device),
            )
            trunc = next_td.get(
                "truncated",
                torch.zeros(*next_td.batch_size, 1, dtype=torch.bool, device=next_td.device),
            )
            next_td = next_td.set("done", term | trunc)
        # Ensure reward/done have trailing dim (1) for SAC/replay compatibility
        for key in ("reward", "done", "terminated"):
            if key in next_td.keys() and next_td[key].ndim == 1:
                next_td = next_td.set(key, next_td[key].unsqueeze(-1))
        flat = flat.set("next", next_td)
    return flat


def _update_step(training: _TrainingSetup, *, device: torch.device, batch_size: int) -> dict[str, float]:
    batch = training.replay.sample(batch_size).to(device)
    training.critic_optimizer.zero_grad()
    critic_loss = training.loss_module(batch)["loss_qvalue"]
    critic_loss.backward()
    training.critic_optimizer.step()

    training.actor_optimizer.zero_grad()
    actor_loss = training.loss_module(batch)["loss_actor"]
    actor_loss.backward()
    training.actor_optimizer.step()

    training.alpha_optimizer.zero_grad()
    alpha_out = training.loss_module(batch)
    alpha_loss = alpha_out["loss_alpha"]
    alpha_loss.backward()
    training.alpha_optimizer.step()

    training.target_updater.step()
    return {
        "loss_actor": float(actor_loss.detach().cpu()),
        "loss_critic": float(critic_loss.detach().cpu()),
        "loss_alpha": float(alpha_loss.detach().cpu()),
        "alpha": float(alpha_out["alpha"].detach().cpu()),
        "entropy": float(alpha_out["entropy"].detach().cpu()),
    }


def _run_sac_eval_log_checkpoint(
    config,
    env,
    modules,
    training,
    state,
    step,
    runtime,
    start_time,
    latest_losses,
    total_updates,
    evaluate_for_best,
):
    def _eval_heldout(cfg, env_setup, local_modules, local_state, *, device, heldout_i_noise=99999):
        return sac_deps.torchrl_sac_loop.evaluate_heldout_if_enabled(
            cfg,
            env_setup,
            local_modules,
            local_state,
            device=device,
            heldout_i_noise=heldout_i_noise,
            capture_actor_state=sac_deps.torchrl_sac_actor_eval.capture_sac_actor_snapshot,
            restore_actor_state=sac_deps.torchrl_sac_actor_eval.restore_sac_actor_snapshot,
            eval_policy_factory=lambda a, e, d: (
                sac_deps.torchrl_sac_actor_eval.SacActorEvalPolicy(
                    a.actor_backbone,
                    a.actor_head,
                    a.obs_scaler,
                    act_dim=e.act_dim,
                    device=d,
                    from_pixels=bool(getattr(e.env_conf, "from_pixels", False)),
                )
            ),
            get_env_conf=sac_deps.get_env_conf,
            evaluate_for_best=evaluate_for_best,
        )

    sac_deps.torchrl_sac_loop.evaluate_if_due(
        config,
        env,
        modules,
        training,
        state,
        step=step,
        device=runtime.device,
        start_time=start_time,
        latest_losses=latest_losses,
        total_updates=total_updates,
        evaluate_actor=_evaluate_actor,
        capture_actor_state=sac_deps.torchrl_sac_actor_eval.capture_sac_actor_snapshot,
        evaluate_heldout=_eval_heldout,
    )
    sac_deps.torchrl_sac_loop.log_if_due(
        config,
        state,
        step=step,
        start_time=start_time,
        latest_losses=latest_losses,
        total_updates=total_updates,
    )
    sac_deps.torchrl_sac_loop.checkpoint_if_due(
        config,
        modules,
        training,
        state,
        step=step,
        build_checkpoint_payload=_checkpoint_payload,
    )


def _process_sac_batch(batch, config, training, runtime, latest_losses, total_updates):
    flat = _flatten_batch_to_transitions(batch)
    n_frames = int(flat.shape[0]) if flat.ndim > 0 else 1
    for i in range(n_frames):
        training.replay.add(flat[i].clone())
    n_update_cycles = max(0, n_frames // int(config.update_every))
    for _ in range(n_update_cycles * int(config.updates_per_step)):
        if training.replay.write_count >= int(config.learning_starts):
            latest_losses = _update_step(training, device=runtime.device, batch_size=int(config.batch_size))
            total_updates += 1
    return latest_losses, total_updates, n_frames


def train_sac(config: SACConfig) -> TrainResult:
    # TorchRL's `ContinuousDistribution.support` currently calls
    # `torch.distributions.constraints.real()` which fails when `validate_args=True`.
    # Keep SAC stable regardless of global validation settings.
    set_gym_backend("gymnasium")
    with sac_deps.torchrl_common.temporary_distribution_validate_args(False):
        if config.eval_noise_mode is not None:
            sac_deps.eval_noise.normalize_eval_noise_mode(config.eval_noise_mode)
        problem_seed = sac_deps.seed_util.resolve_problem_seed(seed=config.seed, problem_seed=config.problem_seed)
        sac_deps.seed_all(sac_deps.seed_util.global_seed_for_run(problem_seed))
        env = build_env_setup(config)
        runtime = config.resolve_runtime(capabilities=_SAC_RUNTIME_CAPABILITIES)
        modules, loss_module = build_modules(config, env, device=runtime.device)
        training = build_training(config, loss_module)
        state = _resume_if_requested(config, modules, training, device=runtime.device)

        from rl.algos import logger as rl_logger

        rl_logger.log_run_header("sac", config, env, training, runtime)

        total_frames = int(config.total_timesteps) - state.start_step
        if total_frames <= 0:
            total_frames = 1
        collector = _build_sac_collector(
            config,
            env,
            modules,
            runtime=runtime,
            total_frames=total_frames,
        )
        collector.set_seed(int(env.problem_seed))
        if hasattr(collector, "shutdown"):
            pass  # multi collectors have shutdown; single Collector may not need it

        start_time = time.time()
        latest_losses = {
            "loss_actor": float("nan"),
            "loss_critic": float("nan"),
            "loss_alpha": float("nan"),
        }
        total_updates = 0
        step = state.start_step

        for batch in collector:
            latest_losses, total_updates, n_frames = _process_sac_batch(batch, config, training, runtime, latest_losses, total_updates)
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
                    runtime,
                    start_time,
                    latest_losses,
                    total_updates,
                    sac_deps.opt_traj.evaluate_for_best,
                )
                break
            _run_sac_eval_log_checkpoint(
                config,
                env,
                modules,
                training,
                state,
                step,
                runtime,
                start_time,
                latest_losses,
                total_updates,
                sac_deps.opt_traj.evaluate_for_best,
            )

        try:
            collector.shutdown()
        except Exception:
            pass
        total_time = time.time() - start_time
        rl_logger.log_run_footer(
            state.best_return,
            int(config.total_timesteps),
            total_time,
            algo_name="sac",
            step_label="steps",
        )
        sac_deps.torchrl_sac_loop.save_final_checkpoint_if_enabled(
            config,
            modules,
            training,
            state,
            build_checkpoint_payload=_checkpoint_payload,
        )

        if config.video_enable:
            ctx = sac_deps.video.RLVideoContext(
                build_eval_env_conf=lambda ps, ns: sac_deps.get_env_conf(config.env_tag, problem_seed=ps, noise_seed_0=ns),
                make_eval_policy=lambda m, d: (
                    sac_deps.torchrl_sac_actor_eval.SacActorEvalPolicy(
                        m.actor_backbone,
                        m.actor_head,
                        m.obs_scaler,
                        act_dim=env.act_dim,
                        device=d,
                    )
                ),
                capture_actor_state=sac_deps.torchrl_sac_actor_eval.capture_sac_actor_snapshot,
                with_actor_state=sac_deps.torchrl_sac_actor_eval.use_sac_actor_snapshot,
            )
            sac_deps.video.render_policy_videos_rl(config, env, modules, training, state, ctx, device=runtime.device)

        return TrainResult(
            best_return=float(state.best_return),
            last_eval_return=float(state.last_eval_return),
            last_heldout_return=state.last_heldout_return,
            num_steps=int(config.total_timesteps),
        )


def register():
    sac_deps.registry.register_algo("sac", SACConfig, train_sac)


__all__ = [
    "SACConfig",
    "TrainResult",
    "_scale_action_to_env",
    "_unscale_action_from_env",
    "register",
    "train_sac",
]
