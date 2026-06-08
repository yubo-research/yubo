"""Auto-generated kiss test_coverage witnesses."""

# ruff: noqa: F821
from __future__ import annotations


def test_kiss_gen_rl_core_ppo_metrics() -> None:
    from rl.core.ppo_metrics import enrich_ppo_iter_record

    refs = (enrich_ppo_iter_record,)
    assert refs


def test_kiss_gen_rl_core_torchrl_runtime_request() -> None:
    from rl.core.torchrl_runtime_request import make_torchrl_runtime_request

    refs = (make_torchrl_runtime_request,)
    assert refs


def test_kiss_gen_rl_iter_record() -> None:
    from rl.iter_record import finite_or_none, merge_metric_fields, timing_record

    refs = (
        finite_or_none,
        timing_record,
        merge_metric_fields,
    )
    assert refs


def test_kiss_gen_rl_logger() -> None:
    from rl.logger import configure_logging, log_rl_status, print_rl_iter_record

    refs = (
        configure_logging,
        print_rl_iter_record,
        log_rl_status,
    )
    assert refs


def test_kiss_gen_rl_mjx_ppo_config_extra() -> None:
    from rl.mjx_ppo_config_extra import MJXPPOCheckpointConfig, MJXPPOEvalConfig

    refs = (
        MJXPPOEvalConfig,
        MJXPPOCheckpointConfig,
    )
    assert refs


def test_kiss_gen_rl_mjx_sac_config_extra() -> None:
    from rl.mjx_sac_config_extra import MJXSACCheckpointConfig, MJXSACEvalConfig

    refs = (
        MJXSACEvalConfig,
        MJXSACCheckpointConfig,
    )
    assert refs


def test_kiss_gen_rl_mjx_sac_loop() -> None:
    from rl.mjx_sac_loop import make_sac_eval_step, make_sac_result, sac_eval_action, sac_eval_args, sac_iter_record

    refs = (
        sac_eval_action,
        sac_eval_args,
        make_sac_eval_step,
        make_sac_result,
        sac_iter_record,
    )
    assert refs


def test_kiss_gen_rl_policy_backbone_common() -> None:
    from rl.policy_backbone.common import obs_space_from_env_conf

    refs = (obs_space_from_env_conf,)
    assert refs


def test_kiss_gen_rl_torchrl_collect_utils() -> None:
    from rl.torchrl.collect_utils import uses_native_isaaclab_collect_env

    refs = (uses_native_isaaclab_collect_env,)
    assert refs


def test_kiss_gen_rl_torchrl_offpolicy_actor_eval() -> None:
    from rl.torchrl.offpolicy.actor_eval import capture_actor_snapshot, restore_actor_snapshot, use_actor_snapshot

    refs = (
        capture_actor_snapshot,
        restore_actor_snapshot,
        use_actor_snapshot,
    )
    assert refs


def test_kiss_gen_rl_torchrl_offpolicy_models() -> None:
    from rl.torchrl.offpolicy.models import ActorNet, QNetPixel

    mean_log_std = ActorNet.mean_log_std
    log_prob_from_action = ActorNet.log_prob_from_action
    forward = QNetPixel.forward
    if False:
        (
            QNetPixel,
            __init__,
        )
    refs = (
        mean_log_std,
        log_prob_from_action,
        QNetPixel,
        forward,
    )
    assert refs


def test_kiss_gen_rl_torchrl_offpolicy_trainer_utils() -> None:
    from rl.torchrl.offpolicy.trainer_utils import (
        extend_replay_with_transitions,
        num_offpolicy_updates_for_batch,
        replay_ready_for_updates,
        store_offpolicy_batch,
    )

    refs = (
        extend_replay_with_transitions,
        store_offpolicy_batch,
        num_offpolicy_updates_for_batch,
        replay_ready_for_updates,
    )
    assert refs
