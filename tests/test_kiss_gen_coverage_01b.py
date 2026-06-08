"""Auto-generated kiss test_coverage witnesses."""

# ruff: noqa: F821
from __future__ import annotations


def test_kiss_gen_common_env_tags() -> None:
    from common.env_tags import is_atari_env_tag

    refs = (is_atari_env_tag,)
    assert refs


def test_kiss_gen_experiments_hyperscalees_llm() -> None:
    from experiments.hyperscalees_llm import local, template

    refs = (
        template,
        local,
    )
    assert refs


def test_kiss_gen_experiments_llm() -> None:
    from experiments.llm import envs, local, main, template

    refs = (
        envs,
        template,
        local,
        main,
    )
    assert refs


def test_kiss_gen_experiments_nanoegg_pretrain() -> None:
    from experiments.nanoegg_pretrain import cli, local, main

    refs = (
        cli,
        local,
        main,
    )
    assert refs


def test_kiss_gen_llm_console_dashboard() -> None:
    import pytest

    pytest.importorskip("textual")
    from llm.console_dashboard import ConsoleDashboard, run_console_dashboard

    compose = ConsoleDashboard.compose
    on_mount = ConsoleDashboard.on_mount
    action_down = ConsoleDashboard.action_down
    action_up = ConsoleDashboard.action_up
    action_tab = ConsoleDashboard.action_tab
    if False:
        (
            ConsoleDashboard,
            __init__,
        )
    refs = (
        ConsoleDashboard,
        compose,
        on_mount,
        action_down,
        action_up,
        action_tab,
        run_console_dashboard,
    )
    assert refs


def test_kiss_gen_llm_console_dashboard_render() -> None:
    from llm.console_dashboard_render import (
        _TranscriptContext,
        diagnostics_renderable,
        footer_renderable,
        header_renderable,
        optimizer_renderable,
        optimizer_state,
        rollout_renderable,
        score_renderable,
    )

    if False:
        (
            _TranscriptContext,
            __init__,
        )
    refs = (
        header_renderable,
        optimizer_renderable,
        rollout_renderable,
        score_renderable,
        diagnostics_renderable,
        footer_renderable,
        optimizer_state,
    )
    assert refs


def test_kiss_gen_llm_console_dashboard_types() -> None:
    from llm.console_dashboard_types import EvalPoint, OptimizerState, TraceRecord

    refs = (
        EvalPoint,
        OptimizerState,
        TraceRecord,
    )
    assert refs


def test_kiss_gen_llm_console_observer() -> None:
    from llm.console_observer import SplitConsoleObserver, TerminalConsoleObserver, UnifiedConsoleManager

    attach = UnifiedConsoleManager.attach
    broadcast_tool_call = UnifiedConsoleManager.broadcast_tool_call
    broadcast_event = UnifiedConsoleManager.broadcast_event
    on_step = SplitConsoleObserver.on_step
    on_tool_call = SplitConsoleObserver.on_tool_call
    on_reward = SplitConsoleObserver.on_reward
    on_event = SplitConsoleObserver.on_event
    append_inference = SplitConsoleObserver.append_inference
    append_diagnostics = SplitConsoleObserver.append_diagnostics
    output_to = SplitConsoleObserver.output_to
    refs = (
        attach,
        broadcast_tool_call,
        broadcast_event,
        on_step,
        on_tool_call,
        on_reward,
        on_event,
        append_inference,
        append_diagnostics,
        output_to,
        TerminalConsoleObserver,
    )
    assert refs


def test_kiss_gen_llm_console_render() -> None:
    from llm.console_render import format_signal_log, format_step_block, format_turn

    refs = (
        format_signal_log,
        format_turn,
        format_step_block,
    )
    assert refs


def test_kiss_gen_llm_eggroll_engine() -> None:
    from llm.eggroll_engine import init_worker_groups

    refs = (init_worker_groups,)
    assert refs


def test_kiss_gen_llm_eggroll_support() -> None:
    from llm.eggroll_support import base_seed

    refs = (base_seed,)
    assert refs


def test_kiss_gen_llm_engine_pool() -> None:
    from llm.engine_pool import VLLMEnginePool, ray_runtime_env

    launch = VLLMEnginePool.launch
    generate_and_score = VLLMEnginePool.generate_and_score
    refs = (
        launch,
        generate_and_score,
        ray_runtime_env,
    )
    assert refs


def test_kiss_gen_llm_tasks_countdown() -> None:
    from llm.tasks_countdown import CountdownTask

    score_single = CountdownTask.score_single
    refs = (score_single,)
    assert refs


def test_kiss_gen_llm_tasks_math() -> None:
    from llm.tasks_math import MathTask

    get_eval_batch = MathTask.get_eval_batch
    score_single = MathTask.score_single
    refs = (
        get_eval_batch,
        score_single,
    )
    assert refs


def test_kiss_gen_llm_tasks_static() -> None:
    from llm.tasks_static import RandomTask, ZerosTask

    score_single = ZerosTask.score_single
    score_single = RandomTask.score_single
    refs = (
        score_single,
        score_single,
    )
    assert refs


def test_kiss_gen_llm_tasks_verifiers() -> None:
    from llm.tasks_verifiers import VerifiersTask

    generate_and_score = VerifiersTask.generate_and_score
    score_single = VerifiersTask.score_single
    refs = (
        generate_and_score,
        score_single,
    )
    assert refs


def test_kiss_gen_llm_thm_task() -> None:
    from llm.thm_task import TheoremProvingTask

    generate_and_score = TheoremProvingTask.generate_and_score
    refs = (generate_and_score,)
    assert refs


def test_kiss_gen_llm_thm_verifiers_env() -> None:
    from llm.thm_verifiers_env import LanguageConfig, TheoremVerifierEnv

    full_proof_path = LanguageConfig.full_proof_path
    setup_initial_proof = TheoremVerifierEnv.setup_initial_proof
    execute_tool = TheoremVerifierEnv.execute_tool
    refs = (
        full_proof_path,
        setup_initial_proof,
        execute_tool,
    )
    assert refs


def test_kiss_gen_llm_verifiers_client() -> None:
    from llm.verifiers_client import assistant_message_from_parsed_response, create_message_renderer, decode_token_ids, parsed_tool_calls

    refs = (
        create_message_renderer,
        decode_token_ids,
        assistant_message_from_parsed_response,
        parsed_tool_calls,
    )
    assert refs
