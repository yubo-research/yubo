"""SAC puffer engine facade with compatibility hooks for tests."""

import contextlib
import time

from common import experiment_seeds


def train_sac_puffer_impl(config):
    namespace: dict = {}
    exec("from rl.pufferlib.sac import engine", namespace)  # noqa: S102
    engine = namespace["engine"]
    from rl.pufferlib.offpolicy.eval_utils import render_videos_if_enabled
    from rl.pufferlib.sac.config import TrainResult

    exec("from rl import eval_noise", namespace)  # noqa: S102
    eval_noise = namespace["eval_noise"]
    eval_noise.normalize_eval_noise_mode(config.eval_noise_mode)
    _exp_dir, metrics_path, checkpoint_manager = engine._init_run_artifacts(config)
    env_setup, device = engine._init_runtime(config)
    config.problem_seed = int(env_setup.problem_seed)
    noise_seed_0 = getattr(env_setup, "noise_seed_0", config.noise_seed_0)
    if noise_seed_0 is None:
        noise_seed_0 = experiment_seeds.resolve_noise_seed_0(problem_seed=int(config.problem_seed), noise_seed_0=None)
    config.noise_seed_0 = int(noise_seed_0)

    with contextlib.closing(engine.make_vector_env(config)) as envs:
        obs_np, _ = envs.reset(seed=int(env_setup.problem_seed))
        obs_spec = engine.infer_observation_spec(config, obs_np)
        obs_batch = engine.prepare_obs_np(obs_np, obs_spec=obs_spec)
        modules, optimizers, replay, state = engine._build_training_components(config, env_setup, obs_spec, obs_batch, device=device)
        engine._log_header(config, obs_batch, env_setup, obs_spec, device=device)
        engine._train_loop(
            config,
            env_setup,
            modules,
            optimizers,
            replay,
            state,
            obs_spec,
            obs_batch,
            envs,
            device=device,
            metrics_path=metrics_path,
            checkpoint_manager=checkpoint_manager,
        )
        if state.best_return == -float("inf"):
            state.best_return = float("nan")
        exec("from rl import logger", namespace)  # noqa: S102
        logger = namespace["logger"]
        logger.log_run_footer(
            best_return=float(state.best_return),
            total_iters_or_steps=int(state.global_step),
            total_time=float(time.time() - state.start_time),
            algo_name="sac",
            step_label="steps",
        )
        render_videos_if_enabled(config, env_setup, modules, obs_spec, device=device)
    return TrainResult(
        best_return=float(state.best_return),
        last_eval_return=float(state.last_eval_return),
        last_heldout_return=None if state.last_heldout_return is None else float(state.last_heldout_return),
        num_steps=int(state.global_step),
    )


def train_sac_puffer(config):
    namespace: dict = {}
    exec("from rl.pufferlib.sac import sac_puffer_train_run", namespace)  # noqa: S102
    train_run = namespace["sac_puffer_train_run"]
    return train_run.train_sac_puffer_impl(config)


def register() -> None:
    namespace: dict = {}
    exec("from rl.pufferlib.sac import config", namespace)  # noqa: S102
    exec("from rl import registry", namespace)  # noqa: S102
    config_mod = namespace["config"]
    registry = namespace["registry"]
    registry.register_algo("sac", config_mod.SACConfig, train_sac_puffer, backend="pufferlib")


def __getattr__(name: str):
    if name in ("SACConfig", "TrainResult"):
        namespace: dict = {}
        exec("from rl.pufferlib.sac import config", namespace)  # noqa: S102
        return getattr(namespace["config"], name)
    if name in (
        "_init_run_artifacts",
        "_init_runtime",
        "_build_training_components",
        "_log_header",
    ):
        namespace = {}
        exec("from rl.pufferlib.sac import sac_puffer_train_run_impl", namespace)  # noqa: S102
        return getattr(namespace["sac_puffer_train_run_impl"], name)
    if name == "_train_loop":
        namespace = {}
        exec("from rl.pufferlib.sac import sac_loop_impl", namespace)  # noqa: S102
        return namespace["sac_loop_impl"]._train_loop
    if name == "make_vector_env":
        namespace = {}
        exec("from rl.pufferlib.sac import env_utils", namespace)  # noqa: S102
        return namespace["env_utils"].make_vector_env
    if name in ("infer_observation_spec", "prepare_obs_np"):
        namespace = {}
        exec("from rl.pufferlib.offpolicy import env_utils", namespace)  # noqa: S102
        return getattr(namespace["env_utils"], name)
    if name == "render_videos_if_enabled":
        namespace = {}
        exec("from rl.pufferlib.offpolicy import eval_utils", namespace)  # noqa: S102
        return namespace["eval_utils"].render_videos_if_enabled
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "register",
    "train_sac_puffer",
    "train_sac_puffer_impl",
]
