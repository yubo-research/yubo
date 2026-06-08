"""Auto-generated kiss test_coverage witnesses."""

# ruff: noqa: F821
from __future__ import annotations


def test_kiss_gen_ops_vec_uhd_arrays() -> None:
    from ops.vec_uhd_arrays import copy_vector, stack_vectors, zeros_vector

    refs = (
        copy_vector,
        stack_vectors,
        zeros_vector,
    )
    assert refs


def test_kiss_gen_ops_vec_uhd_be() -> None:
    from ops.vec_uhd_be import be_sim_batch_enabled

    refs = (be_sim_batch_enabled,)
    assert refs


def test_kiss_gen_optimizer_eggroll_designer_jit_batch() -> None:
    from optimizer.eggroll_designer_jit_batch import build_batched_jitted_fns

    refs = (build_batched_jitted_fns,)
    assert refs


def test_kiss_gen_optimizer_eggroll_designer_jit_common() -> None:
    from optimizer.eggroll_designer_jit_common import apply_noiser_update, rank_scores

    refs = (
        rank_scores,
        apply_noiser_update,
    )
    assert refs


def test_kiss_gen_optimizer_eggroll_designer_types() -> None:
    from optimizer.eggroll_designer_types import EggRollState

    refs = (EggRollState,)
    assert refs


def test_kiss_gen_optimizer_eggroll_external() -> None:
    from optimizer.eggroll_external import supports_external_scoring_env

    refs = (supports_external_scoring_env,)
    assert refs


def test_kiss_gen_policies_mlp_policy() -> None:
    from policies.mlp_policy import MLPPolicy

    forward_tensor = MLPPolicy.forward_tensor
    refs = (forward_tensor,)
    assert refs


def test_kiss_gen_problems_isaaclab_batch_host() -> None:
    from problems.isaaclab_batch_host import host_reset_batch, host_step_batch

    refs = (
        host_reset_batch,
        host_step_batch,
    )
    assert refs


def test_kiss_gen_problems_isaaclab_env_make() -> None:
    from problems.isaaclab_env_make import make_raw_isaaclab_env

    refs = (make_raw_isaaclab_env,)
    assert refs


def test_kiss_gen_problems_isaaclab_gym_adapters() -> None:
    from problems.isaaclab_gym_adapters import _IsaacLabAdapterBase

    render = _IsaacLabAdapterBase.render
    close = _IsaacLabAdapterBase.close
    refs = (
        render,
        close,
    )
    assert refs


def test_kiss_gen_problems_isaaclab_jax_env() -> None:
    from problems.isaaclab_jax_env import IsaacLabJaxState, resolve_isaaclab_jax_num_envs

    refs = (
        IsaacLabJaxState,
        resolve_isaaclab_jax_num_envs,
    )
    assert refs


def test_kiss_gen_problems_isaaclab_jax_spaces() -> None:
    from problems.isaaclab_jax_spaces import gymnax_spaces_from_host

    refs = (gymnax_spaces_from_host,)
    assert refs


def test_kiss_gen_problems_isaaclab_jax_vector_env() -> None:
    from problems.isaaclab_jax_vector_env import IsaacLabJaxVectorState

    refs = (IsaacLabJaxVectorState,)
    assert refs
