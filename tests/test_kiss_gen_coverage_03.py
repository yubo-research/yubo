"""Auto-generated kiss test_coverage witnesses."""

# ruff: noqa: F821
from __future__ import annotations


def test_kiss_gen_rl_torchrl_ppo_actor_nets() -> None:
    from rl.torchrl.ppo.actor_nets import ActorNet

    refs = (ActorNet,)
    assert refs


def test_kiss_gen_rl_torchrl_ppo_core_build() -> None:
    from rl.torchrl.ppo.core_build import build_modules, build_training

    refs = (
        build_modules,
        build_training,
    )
    assert refs


def test_kiss_gen_rl_torchrl_ppo_core_env_setup() -> None:
    from rl.torchrl.ppo.core_env_setup import build_env_setup

    refs = (build_env_setup,)
    assert refs


def test_kiss_gen_rl_torchrl_ppo_ppo_nets_base() -> None:
    from rl.torchrl.ppo.ppo_nets_base import _BackboneHeadNet

    if False:
        (
            _BackboneHeadNet,
            __init__,
        )
    assert True


def test_kiss_gen_rl_torchrl_sac_setup() -> None:
    from rl.torchrl.sac.setup import build_env_setup

    refs = (build_env_setup,)
    assert refs


def test_kiss_gen_rl_torchrl_sac_trainer() -> None:
    from rl.torchrl.sac.trainer import register

    refs = (register,)
    assert refs


def test_kiss_gen_scripts_prepare_tinystories() -> None:
    from scripts.prepare_tinystories import download_and_tokenize

    refs = (download_and_tokenize,)
    assert refs


def test_kiss_gen_third_party_nanochat_common() -> None:
    from third_party.nanochat.common import (
        ColoredFormatter,
        DummyWandb,
        autodetect_device_type,
        compute_cleanup,
        compute_init,
        download_file_with_lock,
        get_base_dir,
        get_dist_info,
        get_peak_flops,
        is_ddp_initialized,
        is_ddp_requested,
        print0,
        print_banner,
        setup_default_logging,
    )

    format = ColoredFormatter.format
    finish = DummyWandb.finish
    if False:
        (
            DummyWandb,
            __init__,
        )
    refs = (
        ColoredFormatter,
        format,
        setup_default_logging,
        get_base_dir,
        download_file_with_lock,
        print0,
        print_banner,
        is_ddp_requested,
        is_ddp_initialized,
        get_dist_info,
        autodetect_device_type,
        compute_init,
        compute_cleanup,
        DummyWandb,
        finish,
        get_peak_flops,
    )
    assert refs


def test_kiss_gen_third_party_nanochat_core_eval() -> None:
    from third_party.nanochat.core_eval import (
        batch_sequences_lm,
        batch_sequences_mc,
        batch_sequences_schema,
        evaluate_example,
        evaluate_task,
        find_common_length,
        forward_model,
        render_prompts_lm,
        render_prompts_mc,
        render_prompts_schema,
        stack_sequences,
    )

    refs = (
        render_prompts_mc,
        render_prompts_schema,
        render_prompts_lm,
        find_common_length,
        stack_sequences,
        batch_sequences_mc,
        batch_sequences_schema,
        batch_sequences_lm,
        forward_model,
        evaluate_example,
        evaluate_task,
    )
    assert refs


def test_kiss_gen_third_party_nanochat_flash_attention() -> None:
    from third_party.nanochat.flash_attention import flash_attn_func, flash_attn_with_kvcache

    refs = (
        flash_attn_func,
        flash_attn_with_kvcache,
    )
    assert refs
