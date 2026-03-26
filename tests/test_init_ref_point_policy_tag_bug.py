"""Tests that EnvironmentRuntime stores env_tag and build_problem requires policy_tag.

_init_ref_point uses env_tag (e.g. "cheetah") rather than env_name (e.g. "HalfCheetah-v5")
for policy-related lookups when callers pass explicit policy_tag.
"""


def test_env_runtime_stores_env_tag_via_build_problem():
    """Test that build_problem correctly stores env_tag in EnvironmentRuntime."""
    from problems.problem import build_problem

    problem = build_problem("cheetah", policy_tag="mlp-32-16", problem_seed=0)
    runtime = problem.env

    assert runtime.env_tag == "cheetah", f"env_tag should be 'cheetah', got '{runtime.env_tag}'"
    assert runtime.env_name == "HalfCheetah-v5", f"env_name should be 'HalfCheetah-v5', got '{runtime.env_name}'"


def test_build_problem_stores_env_tag_with_explicit_policy_tag():
    """env_tag is the logical tag; policy_tag is caller-supplied (not inferred)."""
    from problems.problem import build_problem

    problem = build_problem("cheetah", policy_tag="mlp-32-16", problem_seed=0)
    runtime = problem.env

    assert runtime.env_tag == "cheetah"
    assert problem.policy_tag == "mlp-32-16"


def test_lunar_mlp_with_explicit_policy_tag():
    """lunar-mlp runtime keeps env_tag; policy comes from explicit policy_tag."""
    from problems.problem import build_problem

    problem = build_problem("lunar-mlp", policy_tag="mlp-16-8", problem_seed=0)
    runtime = problem.env

    assert runtime.env_tag == "lunar-mlp"
    assert problem.policy_tag == "mlp-16-8"
