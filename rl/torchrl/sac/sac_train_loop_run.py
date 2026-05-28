from __future__ import annotations


def _resolve_sac_run_seeds(config, t):
    if config.eval.noise_mode is not None:
        t.eval_noise.normalize_eval_noise_mode(config.eval.noise_mode)
    resolved = t.experiment_seeds.resolve_run_seeds(
        seed=int(config.seed),
        problem_seed=config.problem_seed,
        noise_seed_0=config.noise_seed_0,
    )
    config.problem_seed = int(resolved.problem_seed)
    config.noise_seed_0 = int(resolved.noise_seed_0)
    t.seed_all(t.experiment_seeds.global_seed_for_run(int(resolved.problem_seed)))
    return resolved


def _run_sac_training_loop(
    config,
    env,
    modules,
    training,
    state,
    runtime,
    t,
    *,
    _build_sac_collector,
    _process_sac_batch,
    _run_sac_eval_log_checkpoint,
):
    import time

    total_frames = int(config.collector.total_frames) - state.start_step
    if total_frames <= 0:
        total_frames = 1
    collector = _build_sac_collector(config, env, modules, runtime=runtime, total_frames=total_frames)
    collector.set_seed(int(env.problem_seed))
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
            t.phase_b.sync_collector_policy_if_needed(collector, runtime)
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
    return start_time, total_updates, latest_losses


def _finalize_sac_run(config, env, modules, training, state, runtime, start_time, t, *, _checkpoint_payload):
    import time

    total_time = time.time() - start_time
    t.rl_logger.log_run_footer(
        state.best_return,
        int(config.collector.total_frames),
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
