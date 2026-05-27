from __future__ import annotations


def _im(name: str):
    return __import__(name, fromlist=["_"])


def build_sac_collector(
    config,
    env_setup,
    modules,
    *,
    runtime,
    total_frames: int,
):
    st = _im("rl.torchrl.sac.setup")
    _ScaleActionToEnv = st._ScaleActionToEnv
    _make_collect_env_sac = st._make_collect_env_sac

    td_nn = _im("tensordict.nn")
    torch = _im("torch")
    tr_envs = _im("torchrl.envs")
    torchrl_collector_class = _im("rl.core.torchrl_collectors").collector_class
    torchrl_common = _im("rl.core.runtime")
    uses_native_isaaclab_collect_env = _im("rl.torchrl.collect_utils").uses_native_isaaclab_collect_env

    frames_per_batch = int(config.collector.frames_per_batch)
    num_envs = int(config.runtime_num_envs())
    scale_action = td_nn.TensorDictModule(
        _ScaleActionToEnv(env_setup.action_low, env_setup.action_high),
        in_keys=["action"],
        out_keys=["action"],
    ).to(runtime.device)
    collector_policy = td_nn.TensorDictSequential(modules.actor, scale_action)
    if runtime.collector_backend == "single":
        if uses_native_isaaclab_collect_env(env_setup.env_conf):
            vec_env = _make_collect_env_sac(
                env_setup.env_conf,
                env_setup,
                env_index=0,
                num_envs=num_envs,
                device=runtime.device,
            )
        elif num_envs == 1:
            vec_env = _make_collect_env_sac(env_setup.env_conf, env_setup, env_index=0)
        else:
            env_makers = [lambda i=i: _make_collect_env_sac(env_setup.env_conf, env_setup, env_index=i) for i in range(num_envs)]
            vec_env = (
                tr_envs.ParallelEnv(num_envs, env_makers, serial_for_single=True)
                if runtime.single_env_backend == "parallel"
                else tr_envs.SerialEnv(num_envs, env_makers, serial_for_single=True)
            )
        return torchrl_collector_class("Collector")(
            vec_env,
            collector_policy,
            frames_per_batch=frames_per_batch * num_envs,
            total_frames=total_frames,
            init_random_frames=int(config.collector.init_random_frames),
            reset_at_each_iter=False,
            **torchrl_common.collector_device_kwargs(runtime.device),
        )
    num_workers = int(runtime.collector_workers or num_envs)
    create_env_fns = [lambda i=i: _make_collect_env_sac(env_setup.env_conf, env_setup, env_index=i) for i in range(num_workers)]
    frames_per_batch_per_worker = max(1, frames_per_batch)
    collector_cls = torchrl_collector_class("MultiAsyncCollector" if runtime.collector_backend == "multi_async" else "MultiSyncCollector")
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


def update_step(
    config,
    modules,
    training,
    *,
    device,
    batch_size: int,
) -> dict[str, float]:
    st = _im("rl.torchrl.sac.setup")
    sac_update_shared = st.sac_update_shared
    torch = _im("torch")
    batch = training.replay.sample().to(device, non_blocking=device.type == "cuda")
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


def flatten_batch_to_transitions(batch):
    offpolicy_trainer_utils = _im("rl.torchrl.offpolicy.trainer_utils")
    return offpolicy_trainer_utils.flatten_batch_to_transitions(batch)


def normalize_actions_for_replay(flat, *, action_low, action_high):
    offpolicy_trainer_utils = _im("rl.torchrl.offpolicy.trainer_utils")
    return offpolicy_trainer_utils.normalize_actions_for_replay(flat, action_low=action_low, action_high=action_high)


def run_sac_eval_log_checkpoint(
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
    torchrl_sac_loop = _im("rl.torchrl.sac.loop")
    torchrl_sac_actor_eval = _im("rl.torchrl.sac.actor_eval")
    pa = _im("rl.torchrl.sac.sac_trainer_phase_a")
    build_eval_policy = pa.build_eval_policy
    checkpoint_payload = pa.checkpoint_payload
    evaluate_actor = pa.evaluate_actor
    build_problem = _im("problems.problem").build_problem

    def build_env_runtime(
        env_tag,
        *,
        problem_seed=None,
        noise_seed_0=None,
        from_pixels=False,
        pixels_only=True,
    ):
        return build_problem(
            env_tag,
            policy_tag="linear",
            problem_seed=problem_seed,
            noise_seed_0=noise_seed_0,
            from_pixels=from_pixels,
            pixels_only=pixels_only,
        ).env

    def _eval_heldout(cfg, env_setup, local_modules, local_state, *, device, heldout_i_noise=99999):
        return torchrl_sac_loop.evaluate_heldout_if_enabled(
            cfg,
            env_setup,
            local_modules,
            local_state,
            device=device,
            heldout_i_noise=heldout_i_noise,
            capture_actor_state=torchrl_sac_actor_eval.capture_sac_actor_snapshot,
            restore_actor_state=torchrl_sac_actor_eval.restore_sac_actor_snapshot,
            eval_policy_factory=lambda a, e, d: build_eval_policy(a, e, d),
            build_env_runtime=build_env_runtime,
            evaluate_for_best=evaluate_for_best,
        )

    torchrl_sac_loop.evaluate_if_due(
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
        evaluate_actor=evaluate_actor,
        capture_actor_state=torchrl_sac_actor_eval.capture_sac_actor_snapshot,
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
        build_checkpoint_payload=checkpoint_payload,
    )


def process_sac_batch(batch, config, modules, training, runtime, env_setup, latest_losses, total_updates):
    offpolicy_trainer_utils = _im("rl.torchrl.offpolicy.trainer_utils")
    run_chunked_updates = _im("rl.core.update_chunks").run_chunked_updates
    n_frames = offpolicy_trainer_utils.store_offpolicy_batch(batch, training=training, env_setup=env_setup)
    if not offpolicy_trainer_utils.replay_ready_for_updates(training.replay, config):
        return (latest_losses, total_updates, int(n_frames))
    n_updates_due = offpolicy_trainer_utils.num_offpolicy_updates_for_batch(n_frames, config)

    def _run_one_update() -> None:
        nonlocal latest_losses, total_updates
        latest_losses = update_step(
            config,
            modules,
            training,
            device=runtime.device,
            batch_size=int(config.replay_buffer.batch_size),
        )
        total_updates += 1

    run_chunked_updates(n_updates_due, int(config.optim.learner_update_chunk_size), _run_one_update)
    return (latest_losses, total_updates, int(n_frames))


def sync_collector_policy_if_needed(collector, runtime) -> None:
    if runtime.collector_backend == "single":
        return
    update_policy_weights = getattr(collector, "update_policy_weights_", None)
    if callable(update_policy_weights):
        update_policy_weights()


__all__ = [
    "build_sac_collector",
    "flatten_batch_to_transitions",
    "normalize_actions_for_replay",
    "process_sac_batch",
    "run_sac_eval_log_checkpoint",
    "sync_collector_policy_if_needed",
    "update_step",
]
