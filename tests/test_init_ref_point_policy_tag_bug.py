"""Test verifying fix for optimizer._init_ref_point policy tag inference.

FIXED BUG: _init_ref_point now uses env_tag (e.g. "cheetah") instead of
env_name (e.g. "HalfCheetah-v5") for policy inference.

These tests verify the fix works correctly.
"""


def test_env_runtime_stores_env_tag_via_build_problem():
    """Test that build_problem correctly stores env_tag in EnvironmentRuntime."""
    from problems.problem import build_problem

    # Build problem with cheetah env_tag
    problem = build_problem("cheetah", problem_seed=0)
    runtime = problem.env

    # env_tag should be stored, not just env_name
    assert runtime.env_tag == "cheetah", f"env_tag should be 'cheetah', got '{runtime.env_tag}'"
    assert runtime.env_name == "HalfCheetah-v5", f"env_name should be 'HalfCheetah-v5', got '{runtime.env_name}'"


def test_policy_inference_uses_env_tag_not_env_name():
    """Test that policy tag inference works correctly using env_tag."""
    from problems.problem import build_problem, infer_default_policy_tag

    # Build problem - this stores env_tag in runtime
    problem = build_problem("cheetah", problem_seed=0)
    runtime = problem.env

    # Using env_tag should give correct policy
    policy_tag_from_env_tag = infer_default_policy_tag(str(runtime.env_tag))
    assert policy_tag_from_env_tag == "mlp-32-16", f"env_tag '{runtime.env_tag}' should give 'mlp-32-16', got '{policy_tag_from_env_tag}'"

    # Using env_name would give wrong policy (verifying the issue)
    policy_tag_from_env_name = infer_default_policy_tag(str(runtime.env_name))
    assert policy_tag_from_env_name == "linear", f"env_name '{runtime.env_name}' should fallback to 'linear', got '{policy_tag_from_env_name}'"


def test_lunar_mlp_policy_inference():
    """Test that lunar-mlp env infers correct policy via env_tag."""
    from problems.problem import build_problem, infer_default_policy_tag

    problem = build_problem("lunar-mlp", problem_seed=0)
    runtime = problem.env

    # env_tag should be "lunar-mlp"
    assert runtime.env_tag == "lunar-mlp"

    # Using env_tag should give correct policy
    policy_tag = infer_default_policy_tag(str(runtime.env_tag))
    assert policy_tag == "mlp-16-8", f"Expected 'mlp-16-8', got '{policy_tag}'"
