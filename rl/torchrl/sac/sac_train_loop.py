from __future__ import annotations

import importlib

from .sac_train_refs import _sac_training_refs


def train_sac(config):
    import time

    t = _sac_training_refs()
    _SAC_RUNTIME_CAPABILITIES = t.cfg_mod._SAC_RUNTIME_CAPABILITIES
    TrainResult = t.cfg_mod.TrainResult
    _checkpoint_payload = t.phase_a.checkpoint_payload
    _resume_if_requested = t.phase_a.resume_if_requested
    _build_sac_collector = t.phase_b.build_sac_collector
    _process_sac_batch = t.phase_b.process_sac_batch
    _run_sac_eval_log_checkpoint = t.phase_b.run_sac_eval_log_checkpoint
    _sync_collector_policy_if_needed = t.phase_b.sync_collector_policy_if_needed
    build_env_setup = t.setup.build_env_setup
    build_modules = t.setup.build_modules
    build_training = t.setup.build_training

    with t.torchrl_common.temporary_distribution_validate_args(False):
        if config.eval_noise_mode is not None:
            t.eval_noise.normalize_eval_noise_mode(config.eval_noise_mode)
        resolved = t.experiment_seeds.resolve_run_seeds(
            seed=int(config.seed),
            problem_seed=config.problem_seed,
            noise_seed_0=config.noise_seed_0,
        )
        config.problem_seed = int(resolved.problem_seed)
        config.noise_seed_0 = int(resolved.noise_seed_0)
        t.seed_all(t.experiment_seeds.global_seed_for_run(int(resolved.problem_seed)))
        env = build_env_setup(config)
        runtime = config.resolve_runtime(capabilities=_SAC_RUNTIME_CAPABILITIES)
        modules = build_modules(config, env, device=runtime.device)
        training = build_training(config, modules)
        state = _resume_if_requested(config, modules, training, device=runtime.device)

        t.rl_logger.log_run_header("sac", config, env, training, runtime)
        total_frames = int(config.total_timesteps) - state.start_step
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
                    t.episode_rollout.evaluate_for_best,
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
                t.episode_rollout.evaluate_for_best,
            )
        try:
            collector.shutdown()
        except Exception:
            pass
        total_time = time.time() - start_time
        t.rl_logger.log_run_footer(
            state.best_return,
            int(config.total_timesteps),
            total_time,
            algo_name="sac",
            step_label="steps",
        )
        t.torchrl_sac_loop.save_final_checkpoint_if_enabled(
            config,
            modules,
            training,
            state,
            build_checkpoint_payload=_checkpoint_payload,
        )
        if t.video.get_video_settings(config).enable:
            ctx = t.video.RLVideoContext(
                build_eval_env_conf=lambda ps, ns: t.get_env_conf(config.env_tag, problem_seed=ps, noise_seed_0=ns),
                make_eval_policy=lambda m, d: t.phase_a.build_eval_policy(m, env, d),
                capture_actor_state=t.torchrl_sac_actor_eval.capture_sac_actor_snapshot,
                with_actor_state=t.torchrl_sac_actor_eval.use_sac_actor_snapshot,
            )
            t.video.render_policy_videos_rl(config, env, modules, training, state, ctx, device=runtime.device)
        return TrainResult(
            best_return=float(state.best_return),
            last_eval_return=float(state.last_eval_return),
            last_heldout_return=state.last_heldout_return,
            num_steps=int(config.total_timesteps),
        )


def register() -> None:
    registry_module = importlib.import_module("rl.registry")
    SACConfig = importlib.import_module("rl.torchrl.sac.config").SACConfig
    registry_module.register_algo("sac", SACConfig, train_sac)
