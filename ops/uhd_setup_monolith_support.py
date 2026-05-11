from __future__ import annotations

from ops.uhd_setup_util import (
    _action_dim,
    _evaluate_gym_with_denoise,
    _get_device,
    _make_accuracy_fn,
    _maybe_attach_enn,
    _parse_enn_cfg,
    _preload_mnist_train_to_device,
)


def _load_build_problem():
    _ns: dict = {}
    exec("from problems.problem import build_problem", _ns)  # noqa: S102
    return _ns["build_problem"]


def _load_mlp_policy():
    _ns: dict = {}
    exec("from policies.mlp_policy import MLPPolicy", _ns)  # noqa: S102
    return _ns["MLPPolicy"]


def _load_wrap_mlp_env():
    _ns: dict = {}
    exec("from problems.mlp_torch_env import wrap_mlp_env", _ns)  # noqa: S102
    return _ns["wrap_mlp_env"]


def _make_torch_env(problem, **kwargs):
    env_runtime = problem.env
    env_runtime.ensure_spaces()

    policy = problem.build_policy()
    MLPPolicy = _load_mlp_policy()
    if isinstance(policy, MLPPolicy):
        if env_runtime.gym_conf is None:
            raise ValueError("_make_torch_env for MLPPolicy requires a gym_conf with max_steps and num_frames_skip.")
        base_env = env_runtime.make(**kwargs)
        wrap_mlp_env = _load_wrap_mlp_env()
        return wrap_mlp_env(
            env=base_env,
            policy=policy,
            max_steps=env_runtime.gym_conf.max_steps if env_runtime.gym_conf else 1000,
            num_frames_skip=env_runtime.gym_conf.num_frames_skip if env_runtime.gym_conf else 1,
        )

    return env_runtime.make(**kwargs)


__all__ = [
    "_action_dim",
    "_evaluate_gym_with_denoise",
    "_get_device",
    "_load_build_problem",
    "_make_accuracy_fn",
    "_make_torch_env",
    "_maybe_attach_enn",
    "_parse_enn_cfg",
    "_preload_mnist_train_to_device",
]
