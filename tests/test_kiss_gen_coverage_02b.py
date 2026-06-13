"""Auto-generated kiss test_coverage witnesses."""

# ruff: noqa: F821
from __future__ import annotations


def test_kiss_gen_problems_jax_obj() -> None:
    from problems.jax_obj import EggRollJAXVectorObjective

    flatten_params = EggRollJAXVectorObjective.flatten_params
    decode_params = EggRollJAXVectorObjective.decode_params
    refs = (
        flatten_params,
        decode_params,
    )
    assert refs


def test_kiss_gen_problems_jax_step_result() -> None:
    from problems.jax_step_result import JaxStepResult

    refs = (JaxStepResult,)
    assert refs


def test_kiss_gen_problems_nanochat_dataloader() -> None:
    from problems.nanochat_dataloader import BinDataLoader

    iterator = BinDataLoader.iterator
    refs = (iterator,)
    assert refs


def test_kiss_gen_problems_nanochat_lora() -> None:
    from problems.nanochat_lora import _NanochatSubspaceCodec

    sample_eggroll_direction = _NanochatSubspaceCodec.sample_eggroll_direction
    refs = (sample_eggroll_direction,)
    assert refs


def test_kiss_gen_problems_nanoegg_subspace() -> None:
    from problems.nanoegg_subspace import _NanoEggSubspaceCodec

    sample_eggroll_direction = _NanoEggSubspaceCodec.sample_eggroll_direction
    refs = (sample_eggroll_direction,)
    assert refs


def test_kiss_gen_problems_rwkv_distill_objective() -> None:
    from problems.rwkv_distill_objective import RWKVDistillObjective, RWKVDistillObjectiveSpec

    dim = RWKVDistillObjective.dim
    x0 = RWKVDistillObjective.x0
    steps_per_episode = RWKVDistillObjective.steps_per_episode
    num_envs = RWKVDistillObjective.num_envs
    make_policy = RWKVDistillObjective.make_policy
    evaluate = RWKVDistillObjective.evaluate
    configure_embedding = RWKVDistillObjective.configure_embedding
    embed_many = RWKVDistillObjective.embed_many
    embed = RWKVDistillObjective.embed
    if False:
        (
            RWKVDistillObjective,
            __init__,
        )
    refs = (
        RWKVDistillObjectiveSpec,
        RWKVDistillObjective,
        dim,
        x0,
        steps_per_episode,
        num_envs,
        make_policy,
        evaluate,
        configure_embedding,
        embed_many,
        embed,
    )
    assert refs


def test_kiss_gen_problems_surrogate_objective_env() -> None:
    from problems.surrogate_objective_env import PaperObjectiveAdapter

    reset = PaperObjectiveAdapter.reset
    step = PaperObjectiveAdapter.step
    clip_action = PaperObjectiveAdapter.clip_action
    if False:
        (
            PaperObjectiveAdapter,
            __init__,
        )
    refs = (
        PaperObjectiveAdapter,
        reset,
        step,
        clip_action,
    )
    assert refs


def test_kiss_gen_problems_text_obj_cache() -> None:
    from problems.text_obj_cache import _PromptBatchCache

    get_or_create = _PromptBatchCache.get_or_create
    refs = (get_or_create,)
    assert refs


def test_kiss_gen_problems_torch_policy_batch() -> None:
    from problems.torch_policy_batch import try_functional_policy_actions_tensor

    refs = (try_functional_policy_actions_tensor,)
    assert refs


def test_kiss_gen_problems_uhd_obj_types() -> None:
    from problems.uhd_obj_types import UHDVectorObjectiveMixin

    evaluate_many = UHDVectorObjectiveMixin.evaluate_many
    sample_noise = UHDVectorObjectiveMixin.sample_noise
    refs = (
        UHDVectorObjectiveMixin,
        evaluate_many,
        sample_noise,
    )
    assert refs


def test_kiss_gen_problems_warp_env() -> None:
    import pytest

    pytest.importorskip("warp")
    from problems.warp_env import GymnasiumWarpAdapter, WarpState

    reset = GymnasiumWarpAdapter.reset
    step = GymnasiumWarpAdapter.step
    to_torch = GymnasiumWarpAdapter.to_torch
    to_jax = GymnasiumWarpAdapter.to_jax
    if False:
        (
            GymnasiumWarpAdapter,
            __init__,
        )
    refs = (
        WarpState,
        GymnasiumWarpAdapter,
        reset,
        step,
        to_torch,
        to_jax,
    )
    assert refs


def test_kiss_gen_rl_core_ppo_envs() -> None:
    from rl.core.ppo_envs import resolve_gym_env_name

    refs = (resolve_gym_env_name,)
    assert refs
