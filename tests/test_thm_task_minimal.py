import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm.thm_task import LanguageConfig, TheoremProvingTask


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_thm_task_react_loop(monkeypatch):
    # Mock verifiers and prime_sandboxes
    verifiers_types_mod = types.ModuleType("verifiers.types")
    verifiers_types_mod.AssistantMessage = MagicMock
    verifiers_types_mod.ToolCall = MagicMock
    monkeypatch.setitem(sys.modules, "verifiers", types.ModuleType("verifiers"))
    monkeypatch.setitem(sys.modules, "verifiers.types", verifiers_types_mod)

    prime_sandboxes_mod = types.ModuleType("prime_sandboxes")
    prime_sandboxes_mod.CreateSandboxRequest = MagicMock
    prime_sandboxes_mod.AsyncSandboxClient = MagicMock
    monkeypatch.setitem(sys.modules, "prime_sandboxes", prime_sandboxes_mod)

    # Mock Language Config
    lang_cfg = LanguageConfig(
        name="lean4",
        extension="lean",
        docker_image="mock-image",
        compile_cmd="lean {path}",
        guard_begin="-- guard begin",
        guard_end="-- guard end",
        workdir="/workspace",
    )

    task = TheoremProvingTask(batch_size=1, language="lean4")
    task.lang_cfg = lang_cfg  # Ensure we use our mock config

    # Mock LLM and Client
    llm = MagicMock()
    sampling = {}
    lora_spec = None
    answer = {"statement": "theorem test : 1 + 1 = 2 := rfl", "row": {}}

    # Mock Sandbox
    sandbox_client = AsyncMock()
    sandbox = MagicMock()
    sandbox.id = "test-sandbox-id"
    sandbox_client.create.return_value = sandbox

    # Mock LLM response with tool call
    from verifiers.types import AssistantMessage, ToolCall

    # Turn 1: Model thinks and uses tool
    msg1 = AssistantMessage(
        content="I will try to prove this.",
        tool_calls=[
            ToolCall(
                id="call1",
                name="lean4",
                arguments=json.dumps({"code": "theorem test : 1 + 1 = 2 := rfl"}),
            )
        ],
    )

    # Turn 2: Model sees output and says QED
    msg2 = AssistantMessage(content="The compiler was happy. QED")

    responses = [
        MagicMock(choices=[MagicMock(message=msg1)]),
        MagicMock(choices=[MagicMock(message=msg2)]),
    ]

    # Mock _VLLMRLMClient.create
    async def side_effect(*args, **kwargs):
        if responses:
            return responses.pop(0)
        return MagicMock(choices=[MagicMock(message=MagicMock(content="QED", tool_calls=None))])

    mock_client = AsyncMock()
    mock_client.create.side_effect = side_effect
    monkeypatch.setattr("llm.thm_task._VLLMRLMClient", MagicMock(return_value=mock_client))

    # Mock sandbox execute_command to return success
    sandbox_client.execute_command.return_value = MagicMock(stdout="", stderr="", exit_code=0)

    # Mock Console
    task.console = AsyncMock()

    # Run
    reward, log = await task._run_single(llm, "Prove it", sampling, lora_spec, answer, sandbox_client)

    # Assertions
    assert "QED" in log
