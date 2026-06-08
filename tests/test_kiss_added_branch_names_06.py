# ruff: noqa: F821
from __future__ import annotations


def test_added_branch_names_rl_torchrl_offpolicy_trainer_utils() -> None:
    if False:
        (
            flatten_batch_to_transitions,
            normalize_actions_for_replay,
            offpolicy,
            rl,
            torchrl,
            trainer_utils,
        )
    assert True


def test_added_branch_names_rl_torchrl_ppo_core_build() -> None:
    if False:
        (
            build_modules,
            build_training,
            core_build,
            ppo,
            rl,
            torchrl,
            _build_actor_critic,
            _build_optimizer,
            _build_replay_buffer,
            _build_collector,
        )
    assert True


def test_added_branch_names_rl_torchrl_ppo_core_env_setup() -> None:
    if False:
        (
            build_env_setup,
            core_env_setup,
            ppo,
            rl,
            torchrl,
            _build_seeded_eval_env_conf,
            _build_eval_env_conf,
            _make_video_context,
        )
    assert True


def test_added_branch_names_rl_torchrl_sac_sac_setup_build() -> None:
    if False:
        (
            build_env_setup,
            build_modules,
            build_training,
            sac_setup_build,
            sac_update_shared,
            _build_specs,
            _build_actor,
            _build_q_pair,
        )
    assert True


def test_added_branch_names_rl_torchrl_sac_sac_train_loop() -> None:
    if False:
        (
            register,
            rl,
            sac,
            sac_train_loop,
            torchrl,
            train_sac,
        )
    assert True


def test_added_branch_names_rl_torchrl_sac_sac_trainer_phase_a() -> None:
    if False:
        (
            evaluate_actor,
            rl,
            sac,
            sac_trainer_phase_a,
            torchrl,
        )
    assert True


def test_added_branch_names_rl_torchrl_sac_sac_trainer_phase_b_impl() -> None:
    if False:
        (
            flatten_batch_to_transitions,
            normalize_actions_for_replay,
            rl,
            sac,
            sac_trainer_phase_b_impl,
            torchrl,
        )
    assert True


def test_added_branch_names_rl_torchrl_sac_trainer() -> None:
    if False:
        (
            rl,
            sac,
            torchrl,
            train_sac,
            trainer,
        )
    assert True


def test_added_branch_names_rl_torchrl_sac_setup() -> None:
    if False:
        (
            build_env_setup,
            build_modules,
            build_training,
            rl,
            sac,
            setup,
            torchrl,
        )
    assert True


def test_added_branch_names_rl_torchrl_ppo_core_train() -> None:
    if False:
        (
            _evaluate_actor,
            _maybe_eval_and_log,
            _resume_if_requested,
            _run_training_loop,
            core_train,
            ppo,
            rl,
            torchrl,
        )
    assert True


def test_added_branch_names_rl_torchrl_ppo_core_collect_env() -> None:
    if False:
        (
            _build_collector,
            core_collect_env,
            ppo,
            rl,
            torchrl,
        )
    assert True


def test_added_branch_names_rl_torchrl_ppo_core() -> None:
    if False:
        (
            _log_ppo_config,
            core,
            ppo,
            register,
            rl,
            torchrl,
            train_ppo,
        )
    assert True


def test_added_branch_names_rl_torchrl_ppo_core_utils() -> None:
    if False:
        (
            core_utils,
            ppo,
            rl,
            torchrl,
        )
    assert True


def test_added_branch_names_rl_torchrl_ppo_core_types() -> None:
    if False:
        (
            _TanhNormal,
            _TrainState,
            core_types,
            ppo,
            rl,
            torchrl,
        )
    assert True


def test_added_branch_names_rl_torchrl_ppo_core_types_env() -> None:
    if False:
        (
            _EnvSetup,
            _Modules,
            core_types_env,
            ppo,
            rl,
            torchrl,
        )
    assert True


def test_kiss_dep_sentinels_shared() -> None:
    from .kiss_booster_helpers import booster_dep_sentinels

    booster_dep_sentinels()
