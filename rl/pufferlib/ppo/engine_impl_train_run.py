"""PPO puffer training implementation body."""

from __future__ import annotations

from common.im import im


def train_ppo_puffer_impl(config):
    puffer_eval = im("rl.pufferlib.ppo.eval")
    core_env_conf = im("rl.core.env_conf")
    _h = im("rl.pufferlib.ppo.engine_helpers")
    _train = im("rl.pufferlib.ppo.engine_train")
    run_training = _train.run_training
    puffer_eval.validate_eval_config(config)
    resolved = core_env_conf.resolve_run_seeds(
        seed=int(config.seed),
        problem_seed=config.problem_seed,
        noise_seed_0=config.noise_seed_0,
    )
    config.problem_seed = int(resolved.problem_seed)
    config.noise_seed_0 = int(resolved.noise_seed_0)
    device = _h._resolve_device(config.device)
    _h._seed_everything(int(core_env_conf.global_seed_for_run(int(config.problem_seed))))
    plan = _h._build_plan(config)
    metrics_path = _h._prepare_outputs(config)
    closing = im("contextlib").closing
    with closing(_h.make_vector_env(config)) as envs:
        return run_training(
            config,
            plan,
            device,
            metrics_path,
            envs,
            build_eval_env_conf_fn=_h.build_eval_env_conf,
            init_runtime_fn=_h._init_runtime,
            prepare_obs_fn=_h._prepare_obs,
        )


def train_ppo_puffer(config):
    return train_ppo_puffer_impl(config)
