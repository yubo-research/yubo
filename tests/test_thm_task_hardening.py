from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm.episodes import RuntimeConfig
from llm.thm_task import LanguageConfig
from llm.thm_verifiers_env import TheoremVerifierEnv
from tests.test_thm_task_minimal import _proof_case, _proof_episode, _ProofSandbox
from tests.thm_test_helpers import lean_proof_file_executor

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _DeleteFailSandbox(_ProofSandbox):
    async def delete(self, sandbox_id):
        raise RuntimeError("delete failed")


async def test_proof_episode_rejects_prose_without_compiling():
    episode, sandbox = _proof_episode(["The problem is to prove True. I will explain it in words."], max_turns=1)

    signal = await episode.run(_proof_case(), MagicMock(), {}, RuntimeConfig())

    assert signal.reward == 0.0
    assert signal.status == "bad_candidate"
    assert signal.metrics["compile_calls"] == 0
    assert sandbox.compile_calls == 0


async def test_proof_episode_cleanup_failure_does_not_mask_result():
    signal = await _run_cleanup_failure_case()

    assert signal.reward == 1.0
    assert signal.status == "ok"


async def _run_cleanup_failure_case():
    episode, _sandbox = _proof_episode(
        ["theorem test : True := by\n  trivial"],
        sandbox=_DeleteFailSandbox(),
        max_turns=1,
    )
    return await episode.run(_proof_case(), MagicMock(), {}, RuntimeConfig())


async def test_lean_rubric_ignores_expected_statement_in_comments(monkeypatch):
    verifiers_types_mod = types.ModuleType("verifiers.types")
    verifiers_types_mod.AssistantMessage = MagicMock
    verifiers_types_mod.ToolCall = MagicMock
    monkeypatch.setitem(sys.modules, "verifiers", types.ModuleType("verifiers"))
    monkeypatch.setitem(sys.modules, "verifiers.types", verifiers_types_mod)

    lang_cfg = LanguageConfig(
        name="lean4",
        extension="lean",
        docker_image="mock-image",
        compile_cmd="lean {path}",
        guard_begin="-- guard begin",
        guard_end="-- guard end",
        workdir="/workspace",
        proof_path="tmp/proof",
    )
    env = TheoremVerifierEnv(lang_cfg, max_turns=1)
    proof_file = {"content": "-- theorem test : True := by\n--   trivial\n\ntheorem other : True := by\n  trivial\n"}
    sandbox_client = AsyncMock()
    sandbox_client.execute_command.side_effect = lean_proof_file_executor(proof_file)

    reward = await env.rubric.score_state(
        {
            "sandbox_client": sandbox_client,
            "sandbox_id": "test-sandbox-id",
            "expected_statement": "theorem test : True := by\n  sorry",
        }
    )

    assert reward == 0.0


def test_lean_prompt_formatter_uses_raw_completion_prompt():
    env = TheoremVerifierEnv(
        LanguageConfig(
            name="lean4",
            extension="lean",
            docker_image="mock-image",
            compile_cmd="lean {path}",
            guard_begin="-- guard begin",
            guard_end="-- guard end",
            workdir="/workspace",
            proof_path="tmp/proof",
        )
    )

    prompt = env.user_prompt(
        "ignored prose prompt",
        {"statement": "theorem test : True := by\n  sorry"},
    )
    rendered = env.format_messages_for_generation(
        [
            {"role": "system", "content": env.system_prompt()},
            {"role": "user", "content": prompt},
        ]
    )

    assert rendered is not None
    assert rendered.startswith("Complete the following Lean 4 code:")
    assert "System:" not in rendered
    assert "User:" not in rendered
    assert "set_option maxHeartbeats 0" in rendered
