from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torchrl.collectors as tr_collectors

import numpy as np
import torch

from rl.torchrl.offpolicy import trainer_utils

from . import deps, sac_trainer_phase_b_impl
from .config import _SAC_RUNTIME_CAPABILITIES, SACConfig, TrainResult


def _build_env_runtime(
    env_tag,
    *,
    problem_seed=None,
    noise_seed_0=None,
    from_pixels=False,
    pixels_only=True,
):
    """Wrapper that adapts build_problem to return EnvironmentRuntime for callback-based APIs."""
    return deps.build_problem(
        env_tag,
        policy_tag="linear",
        problem_seed=problem_seed,
        noise_seed_0=noise_seed_0,
        from_pixels=from_pixels,
        pixels_only=pixels_only,
    ).env


def _setup_attr(name: str):
    from . import setup

    return getattr(setup, name)


def _phase_a_attr(name: str):
    from . import sac_trainer_phase_a

    return getattr(sac_trainer_phase_a, name)


def _build_eval_policy(modules, env_setup, device: torch.device):
    return deps.torchrl_sac_actor_eval.SacActorEvalPolicy(
        modules.actor_backbone,
        modules.actor_head,
        modules.obs_scaler,
        act_dim=env_setup.act_dim,
        device=device,
        from_pixels=bool(getattr(env_setup.env_conf, "from_pixels", False)),
    )


def _evaluate_actor(
    config: SACConfig,
    env,
    modules,
    *,
    device: torch.device,
    eval_seed: int,
) -> float:
    from rl.core.episode_rollout import collect_denoised_trajectory

    eval_env = env.env_conf
    eval_policy = _build_eval_policy(modules, env, device)
    traj, _ = collect_denoised_trajectory(
        eval_env,
        eval_policy,
        num_denoise=config.eval.num_denoise,
        i_noise=int(eval_seed),
    )
    return float(traj.rreturn)


def _build_sac_collector(
    config: SACConfig,
    env_setup,
    modules,
    *,
    runtime,
    total_frames: int,
) -> tr_collectors.Collector | tr_collectors.MultiSyncCollector | tr_collectors.MultiAsyncCollector:
    import torchrl.collectors as tr_collectors
    import torchrl.envs as tr_envs

    _make_collect_env_sac = _setup_attr("_make_collect_env_sac")
    _ScaleActionToEnv = _setup_attr("_ScaleActionToEnv")
    frames_per_batch = int(config.collector.frames_per_batch)
    num_envs = int(config.runtime_num_envs())
    scale_action = deps.td_nn.TensorDictModule(
        _ScaleActionToEnv(env_setup.action_low, env_setup.action_high),
        in_keys=["action"],
        out_keys=["action"],
    ).to(runtime.device)
    collector_policy = deps.td_nn.TensorDictSequential(modules.actor, scale_action)
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
            init_random_frames=int(config.collector.init_random_frames),
            reset_at_each_iter=False,
            **deps.torchrl_common.collector_device_kwargs(runtime.device),
        )
    num_workers = int(runtime.collector_workers or num_envs)
    create_env_fns = [lambda i=i: _make_collect_env_sac(env_setup.env_conf, env_setup, env_index=i) for i in range(num_workers)]
    frames_per_batch_per_worker = max(1, frames_per_batch)
    collector_cls = tr_collectors.MultiAsyncCollector if runtime.collector_backend == "multi_async" else tr_collectors.MultiSyncCollector
    return collector_cls(
        create_env_fn=create_env_fns,
        policy=collector_policy,
        frames_per_batch=[int(frames_per_batch_per_worker)] * num_workers,
        total_frames=total_frames,
        init_random_frames=int(config.collector.init_random_frames),
        reset_at_each_iter=False,
        env_device=torch.device("cpu"),
        policy_device=runtime.device,
        storing_device=torch.device("cpu"),
    )


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
    runtime,
    start_time,
    latest_losses,
    total_updates,
    evaluate_for_best,
):
    def _eval_heldout(cfg, env_setup, local_modules, local_state, *, device, heldout_i_noise=99999):
        return deps.torchrl_sac_loop.evaluate_heldout_if_enabled(
            cfg,
            env_setup,
            local_modules,
            local_state,
            device=device,
            heldout_i_noise=heldout_i_noise,
            capture_actor_state=deps.torchrl_sac_actor_eval.capture_sac_actor_snapshot,
            restore_actor_state=deps.torchrl_sac_actor_eval.restore_sac_actor_snapshot,
            eval_policy_factory=lambda a, e, d: _build_eval_policy(a, e, d),
            build_env_runtime=_build_env_runtime,
            evaluate_for_best=evaluate_for_best,
        )

    deps.torchrl_sac_loop.evaluate_if_due(
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
        capture_actor_state=deps.torchrl_sac_actor_eval.capture_sac_actor_snapshot,
        evaluate_heldout=_eval_heldout,
    )
    deps.torchrl_sac_loop.log_if_due(
        config,
        state,
        step=step,
        start_time=start_time,
        latest_losses=latest_losses,
        total_updates=total_updates,
    )
    deps.torchrl_sac_loop.checkpoint_if_due(
        config,
        modules,
        training,
        state,
        step=step,
        build_checkpoint_payload=_phase_a_attr("checkpoint_payload"),
    )


def _process_sac_batch(batch, config, modules, training, runtime, env_setup, latest_losses, total_updates):
    return sac_trainer_phase_b_impl.process_sac_batch(
        batch,
        config,
        modules,
        training,
        runtime,
        env_setup,
        latest_losses,
        total_updates,
    )


def _sync_collector_policy_if_needed(collector, runtime) -> None:
    sac_trainer_phase_b_impl.sync_collector_policy_if_needed(collector, runtime)


def train_sac(config: SACConfig) -> TrainResult:
    build_env_setup = _setup_attr("build_env_setup")
    build_modules = _setup_attr("build_modules")
    build_training = _setup_attr("build_training")
    sac_resume_if_requested = _phase_a_attr("resume_if_requested")

    with deps.torchrl_common.temporary_distribution_validate_args(False):
        if config.eval.noise_mode is not None:
            deps.eval_noise.normalize_eval_noise_mode(config.eval.noise_mode)
        resolved = deps.experiment_seeds.resolve_run_seeds(
            seed=int(config.seed),
            problem_seed=config.problem_seed,
            noise_seed_0=config.noise_seed_0,
        )
        config.problem_seed = int(resolved.problem_seed)
        config.noise_seed_0 = int(resolved.noise_seed_0)
        deps.seed_all(deps.experiment_seeds.global_seed_for_run(int(resolved.problem_seed)))
        env = build_env_setup(config)
        runtime = config.resolve_runtime(capabilities=_SAC_RUNTIME_CAPABILITIES)
        modules = build_modules(config, env, device=runtime.device)
        training = build_training(config, modules)
        state = sac_resume_if_requested(config, modules, training, device=runtime.device)
        from rl import logger

        logger.log_run_header("sac", config, env, training, runtime)
        total_frames = int(config.collector.total_frames) - state.start_step
        if total_frames <= 0:
            total_frames = 1
        collector = _build_sac_collector(config, env, modules, runtime=runtime, total_frames=total_frames)
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
            updates_before_batch = int(total_updates)
            latest_losses, total_updates, n_frames = _process_sac_batch(
                batch,
                config,
                modules,
                training,
                runtime,
                env,
                latest_losses,
                total_updates,
            )
            if int(total_updates) > updates_before_batch:
                _sync_collector_policy_if_needed(collector, runtime)
            step += n_frames
            if step >= int(config.collector.total_frames):
                step = int(config.collector.total_frames)
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
                    deps.episode_rollout.evaluate_for_best,
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
                deps.episode_rollout.evaluate_for_best,
            )
        try:
            collector.shutdown()
        except Exception:
            pass
        total_time = time.time() - start_time
        logger.log_run_footer(
            state.best_return,
            int(config.collector.total_frames),
            total_time,
            algo_name="sac",
            step_label="steps",
        )
        deps.torchrl_sac_loop.save_final_checkpoint_if_enabled(
            config,
            modules,
            training,
            state,
            build_checkpoint_payload=_phase_a_attr("checkpoint_payload"),
        )
        if deps.video.get_video_settings(config).enable:
            ctx = deps.video.RLVideoContext(
                build_eval_env_conf=lambda ps, ns: _build_env_runtime(config.env_tag, problem_seed=ps, noise_seed_0=ns),
                make_eval_policy=lambda m, d: _build_eval_policy(m, env, d),
                capture_actor_state=deps.torchrl_sac_actor_eval.capture_sac_actor_snapshot,
                with_actor_state=deps.torchrl_sac_actor_eval.use_sac_actor_snapshot,
            )
            deps.video.render_policy_videos_rl(config, env, modules, training, state, ctx, device=runtime.device)
        return TrainResult(
            best_return=float(state.best_return),
            last_eval_return=float(state.last_eval_return),
            last_heldout_return=state.last_heldout_return,
            num_steps=int(config.collector.total_frames),
        )


def register():
    deps.registry.register_algo("sac", SACConfig, train_sac)


__all__ = [
    "SACConfig",
    "TrainResult",
    "register",
    "train_sac",
]
