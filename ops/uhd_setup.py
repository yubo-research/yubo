"""UHD setup helpers used by kiss-coverage and legacy `ops.uhd_setup` monkeypatch targets."""

from __future__ import annotations

from ops.uhd_config import BEConfig
from ops.uhd_setup_monolith_bszo import _run_bszo_iterations, run_bszo_loop
from ops.uhd_setup_monolith_make_loop import _make_simple_loop_for_np_policy, make_loop
from ops.uhd_setup_monolith_opt import (
    _gym_embed_bounds,
    _make_perturbator,
    _make_simple_optimizer,
    _mnist_embed_bounds,
)
from ops.uhd_setup_monolith_simple_run import _run_simple_gym as _run_simple_gym_impl
from ops.uhd_setup_monolith_simple_run import (
    _run_simple_torch as _run_simple_torch_impl,
)
from ops.uhd_setup_monolith_support import (
    _action_dim,
    _evaluate_gym_with_denoise,
    _get_device,
    _load_build_problem,
    _load_mlp_policy,
    _load_wrap_mlp_env,
    _make_accuracy_fn,
    _make_torch_env,
    _maybe_attach_enn,
    _parse_enn_cfg,
    _preload_mnist_train_to_device,
)


def _run_simple_torch(*args, **kwargs):
    return _run_simple_torch_impl(*args, **kwargs)


def _run_simple_gym(*args, **kwargs):
    return _run_simple_gym_impl(*args, **kwargs)


def run_simple_loop(
    env_tag: str,
    num_rounds: int,
    sigma: float = 0.001,
    optimizer: str = "simple",
    *,
    policy_tag: str | None = None,
    num_dim_target: float | None = None,
    problem_seed: int | None = None,
    noise_seed_0: int | None = None,
    batch_size: int = 4096,
    log_interval: int = 1,
    accuracy_interval: int = 1000,
    target_accuracy: float | None = None,
    num_denoise: int | None = None,
    be: BEConfig | None = None,
) -> None:
    import ops.uhd_setup_monolith_support as sup
    from common.seed_all import seed_all

    if policy_tag is None:
        policy_tag = "pure-function"

    build_problem = sup._load_build_problem()
    problem = build_problem(env_tag, policy_tag, problem_seed=problem_seed, noise_seed_0=noise_seed_0)
    env_runtime = problem.env
    if env_runtime.problem_seed is not None:
        seed_all(int(env_runtime.problem_seed))
    env = env_runtime.make()
    if hasattr(env, "torch_env"):
        _run_simple_torch(
            env,
            env_runtime,
            env_tag,
            num_rounds,
            optimizer=optimizer,
            sigma=sigma,
            num_dim_target=num_dim_target,
            batch_size=batch_size,
            log_interval=log_interval,
            accuracy_interval=accuracy_interval,
            target_accuracy=target_accuracy,
            num_denoise=num_denoise,
            be=be,
        )
    else:
        _run_simple_gym(
            env,
            problem,
            env_tag,
            num_rounds,
            optimizer=optimizer,
            sigma=sigma,
            num_dim_target=num_dim_target,
            log_interval=log_interval,
            target_accuracy=target_accuracy,
            num_denoise=num_denoise,
            be=be,
        )


__all__ = [
    "BEConfig",
    "_action_dim",
    "_evaluate_gym_with_denoise",
    "_get_device",
    "_gym_embed_bounds",
    "_load_build_problem",
    "_load_mlp_policy",
    "_load_wrap_mlp_env",
    "_make_accuracy_fn",
    "_make_perturbator",
    "_make_simple_loop_for_np_policy",
    "_make_simple_optimizer",
    "_make_torch_env",
    "_maybe_attach_enn",
    "_mnist_embed_bounds",
    "_parse_enn_cfg",
    "_preload_mnist_train_to_device",
    "_run_bszo_iterations",
    "_run_simple_gym",
    "_run_simple_torch",
    "make_loop",
    "run_bszo_loop",
    "run_simple_loop",
]
