"""Tests for problems/problem.py module."""

from __future__ import annotations

import pytest


def test_problem_construction():
    from problems.environment_spec import EnvironmentSpec, materialize_env
    from problems.problem import Problem

    spec = EnvironmentSpec(env_name="f:sphere-2d")
    runtime = materialize_env(spec, problem_seed=0)
    problem = Problem(runtime, "pure-function")

    assert problem.env is runtime
    assert problem.policy_tag == "pure-function"


def test_problem_env_property():
    from problems.environment_spec import EnvironmentSpec, materialize_env
    from problems.problem import Problem

    spec = EnvironmentSpec(env_name="f:ackley-3d")
    runtime = materialize_env(spec, problem_seed=42)
    problem = Problem(runtime, "pure-function")

    assert problem.env.problem_seed == 42
    assert problem.env.env_name == "f:ackley-3d"


def test_problem_build_policy_pure_function():
    from problems.problem import build_problem

    problem = build_problem("f:sphere-2d", policy_tag="pure-function", problem_seed=0)
    policy = problem.build_policy()
    assert policy is not None
    assert hasattr(policy, "num_params")
    assert policy.num_params() == 2


def test_resolve_rl_model_defaults_explicit_override():
    from problems.problem import resolve_rl_model_defaults

    cfg = resolve_rl_model_defaults("cheetah", policy_tag="mlp-32-16", algo="sac")
    assert cfg["backbone_hidden_sizes"] == (256, 256)
    assert cfg["backbone_activation"] == "relu"


def test_resolve_rl_model_defaults_from_preset():
    from problems.problem import resolve_rl_model_defaults

    cfg = resolve_rl_model_defaults("f:sphere-2d", policy_tag="mlp-16-8", algo="ppo")
    assert cfg["backbone_hidden_sizes"] == (16, 8)
    assert cfg["backbone_name"] == "mlp"


def test_resolve_rl_model_defaults_missing_policy_tag_raises():
    from problems.problem import resolve_rl_model_defaults

    with pytest.raises(ValueError, match="Missing required argument 'policy_tag'"):
        resolve_rl_model_defaults("cheetah", algo="ppo")


def test_resolve_rl_model_defaults_invalid_algo():
    from problems.problem import resolve_rl_model_defaults

    with pytest.raises(ValueError, match="Unsupported algo"):
        resolve_rl_model_defaults("cheetah", policy_tag="mlp-32-16", algo="invalid")


def test_resolve_rl_model_defaults_no_defaults_error():
    from problems.problem import resolve_rl_model_defaults

    with pytest.raises(ValueError, match="No RL model defaults"):
        resolve_rl_model_defaults("f:sphere-2d", policy_tag="linear", algo="ppo")


def test_resolve_rl_model_defaults_quadruped():
    from problems.problem import resolve_rl_model_defaults

    cfg = resolve_rl_model_defaults("dm_control/quadruped-run-v0", policy_tag="mlp-64-64", algo="ppo")
    assert cfg["backbone_hidden_sizes"] == (64, 64)
    assert cfg["backbone_layer_norm"] is True


def test_build_problem_pure_function():
    from problems.problem import build_problem

    problem = build_problem("f:sphere-2d", policy_tag="pure-function", problem_seed=123)
    assert problem.policy_tag == "pure-function"
    assert problem.env.problem_seed == 123


def test_build_problem_explicit_policy_tag():
    from problems.problem import build_problem

    problem = build_problem("f:sphere-2d", policy_tag="pure-function", problem_seed=0)
    assert problem.policy_tag == "pure-function"


def test_build_problem_with_noise():
    from problems.problem import build_problem

    problem = build_problem(
        "f:sphere-2d",
        policy_tag="pure-function",
        problem_seed=0,
        noise_level=0.1,
        noise_seed_0=42,
        frozen_noise=True,
    )
    assert problem.env.noise_level == 0.1
    assert problem.env.noise_seed_0 == 42
    assert problem.env.frozen_noise is True


def test_build_problem_from_pixels_override():
    from problems.problem import build_problem

    problem = build_problem("f:sphere-2d", policy_tag="pure-function", from_pixels=True, pixels_only=False)
    assert problem.env.spec.from_pixels is True
    assert problem.env.spec.pixels_only is False


def test_build_problem_missing_policy_tag_raises():
    from problems.problem import build_problem

    with pytest.raises(ValueError, match="Missing required argument 'policy_tag'"):
        build_problem("lunar-mlp")


def test_build_problem_returns_problem_instance():
    from problems.problem import Problem, build_problem

    problem = build_problem("f:ackley-3d", policy_tag="pure-function")
    assert isinstance(problem, Problem)


def test_problem_build_policy_ensures_spaces():
    from problems.problem import build_problem

    problem = build_problem("f:sphere-2d", policy_tag="pure-function", problem_seed=0)
    assert problem.env.state_space is None
    policy = problem.build_policy()
    assert policy is not None


def test_problem_rl_model_overrides_coverage():
    from problems.problem import _PROBLEM_RL_MODEL_OVERRIDES

    assert ("cheetah", "mlp-32-16", "ppo") in _PROBLEM_RL_MODEL_OVERRIDES
    assert ("cheetah", "mlp-32-16", "sac") in _PROBLEM_RL_MODEL_OVERRIDES
    assert ("dm_control/quadruped-run-v0", "mlp-64-64", "ppo") in _PROBLEM_RL_MODEL_OVERRIDES


def test_resolve_rl_model_defaults_returns_copy():
    from problems.problem import resolve_rl_model_defaults

    cfg1 = resolve_rl_model_defaults("cheetah", policy_tag="mlp-32-16", algo="ppo")
    cfg2 = resolve_rl_model_defaults("cheetah", policy_tag="mlp-32-16", algo="ppo")
    cfg1["modified"] = True
    assert "modified" not in cfg2
