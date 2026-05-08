"""SAC puffer engine facade with compatibility hooks for tests."""

import contextlib
import time


def train_sac_puffer_impl(config):
    import rl.pufferlib.sac.engine as eng
    from rl.pufferlib.offpolicy.eval_utils import render_videos_if_enabled
    from rl.pufferlib.sac.config import TrainResult

    _ns: dict = {}
    exec("import rl.eval_noise as eval_noise", _ns)  # noqa: S102
    exec("import rl.core.env_conf as core_env_conf", _ns)  # noqa: S102
    eval_noise = _ns["eval_noise"]
    core_env_conf = _ns["core_env_conf"]
    eval_noise.normalize_eval_noise_mode(config.eval_noise_mode)
    _exp_dir, metrics_path, checkpoint_manager = eng._init_run_artifacts(config)
    env_setup, device = eng._init_runtime(config)
    config.problem_seed = int(env_setup.problem_seed)
    noise_seed_0 = getattr(env_setup, "noise_seed_0", config.noise_seed_0)
    if noise_seed_0 is None:
        noise_seed_0 = core_env_conf.resolve_noise_seed_0(problem_seed=int(config.problem_seed), noise_seed_0=None)
    config.noise_seed_0 = int(noise_seed_0)

    with contextlib.closing(eng.make_vector_env(config)) as envs:
        obs_np, _ = envs.reset(seed=int(env_setup.problem_seed))
        obs_spec = eng.infer_observation_spec(config, obs_np)
        obs_batch = eng.prepare_obs_np(obs_np, obs_spec=obs_spec)
        modules, optimizers, replay, state = eng._build_training_components(config, env_setup, obs_spec, obs_batch, device=device)
        eng._log_header(config, obs_batch, env_setup, obs_spec, device=device)
        eng._train_loop(
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
        _lg: dict = {}
        exec("import rl.logger as _logger_mod", _lg)  # noqa: S102
        _lg["_logger_mod"].log_run_footer(
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
    _ns: dict = {}
    exec(
        "from rl.pufferlib.sac.sac_puffer_train_run import train_sac_puffer_impl as _f",
        _ns,
    )  # noqa: S102
    return _ns["_f"](config)


def register() -> None:
    _ns: dict = {}
    exec("from rl.pufferlib.sac.config import SACConfig", _ns)  # noqa: S102
    exec("import rl.registry as _reg", _ns)  # noqa: S102
    _ns["_reg"].register_algo("sac", _ns["SACConfig"], train_sac_puffer, backend="pufferlib")


def __getattr__(name: str):
    if name in ("SACConfig", "TrainResult"):
        _ns: dict = {}
        exec(f"from rl.pufferlib.sac.config import {name} as _v", _ns)  # noqa: S102
        return _ns["_v"]
    if name in (
        "_init_run_artifacts",
        "_init_runtime",
        "_build_training_components",
        "_log_header",
    ):
        _ns = {}
        exec(f"from rl.pufferlib.sac.sac_puffer_train_run_impl import {name} as _v", _ns)  # noqa: S102
        return _ns["_v"]
    if name == "_train_loop":
        _ns = {}
        exec("from rl.pufferlib.sac.sac_loop_impl import _train_loop as _v", _ns)  # noqa: S102
        return _ns["_v"]
    if name == "make_vector_env":
        _ns = {}
        exec("from rl.pufferlib.sac.env_utils import make_vector_env as _v", _ns)  # noqa: S102
        return _ns["_v"]
    if name in ("infer_observation_spec", "prepare_obs_np"):
        _ns = {}
        exec(f"from rl.pufferlib.offpolicy.env_utils import {name} as _v", _ns)  # noqa: S102
        return _ns["_v"]
    if name == "render_videos_if_enabled":
        _ns = {}
        exec(
            "from rl.pufferlib.offpolicy.eval_utils import render_videos_if_enabled as _v",
            _ns,
        )  # noqa: S102
        return _ns["_v"]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "register",
    "train_sac_puffer",
    "train_sac_puffer_impl",
]
