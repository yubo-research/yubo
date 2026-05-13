import sys
import types
from unittest.mock import MagicMock

import pytest

from llm.tasks_verifiers import _ENV_CACHE, VerifiersTask
from tests.llm_test_mocks_env import FakeEnv
from tests.llm_test_mocks_state import FakeAssistantMessage, FakeState
from tests.llm_test_mocks_vllm import FakeRLM


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_verifiers_task_generate_and_score_async(monkeypatch):
    _ENV_CACHE.clear()

    # 1. Mock verifiers modules
    verifiers_mod = types.ModuleType("verifiers")
    verifiers_v1_mod = types.ModuleType("verifiers.v1")
    verifiers_utils_mod = types.ModuleType("verifiers.utils")
    verifiers_env_utils_mod = types.ModuleType("verifiers.utils.env_utils")
    verifiers_types_mod = types.ModuleType("verifiers.types")

    verifiers_types_mod.State = FakeState
    verifiers_types_mod.AssistantMessage = FakeAssistantMessage
    verifiers_types_mod.UserMessage = lambda content: {
        "role": "user",
        "content": content,
    }
    verifiers_types_mod.SystemMessage = lambda content: {
        "role": "system",
        "content": content,
    }
    verifiers_types_mod.ToolMessage = lambda content, tool_call_id: {
        "role": "tool",
        "content": content,
        "tool_call_id": tool_call_id,
    }
    verifiers_types_mod.ToolCall = lambda id, name, arguments: MagicMock(id=id, name=name, arguments=arguments)

    verifiers_v1_mod.RLM = FakeRLM
    verifiers_utils_mod.__path__ = []

    verifiers_env_utils_mod.load_environment = lambda env_id, **kwargs: FakeEnv()

    monkeypatch.setitem(sys.modules, "verifiers", verifiers_mod)
    monkeypatch.setitem(sys.modules, "verifiers.v1", verifiers_v1_mod)
    monkeypatch.setitem(sys.modules, "verifiers.utils", verifiers_utils_mod)
    monkeypatch.setitem(sys.modules, "verifiers.utils.env_utils", verifiers_env_utils_mod)
    monkeypatch.setitem(sys.modules, "verifiers.types", verifiers_types_mod)

    # 2. Mock vLLM
    vllm_mod = types.ModuleType("vllm")
    vllm_mod.SamplingParams = lambda **kwargs: MagicMock(**kwargs)
    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)

    llm = MagicMock()

    async def fake_generate(prompt, sampling_params, request_id, lora_request=None):
        yield MagicMock(outputs=[MagicMock(text="reasoning ```python\nprint(2+2)\n```")])

    llm.generate = fake_generate

    # 3. Initialize task
    task = VerifiersTask(env_id="gsm8k", batch_size=1)

    # 4. Run generate_and_score_async
    fitnesses, info, logs = await task.generate_and_score_async(
        llm=llm,
        prompts=["2+2?"],
        sampling_params_kwargs={"temperature": 0.0},
        lora_request_specs=None,
        answers=[{"__verifiers_payload__": True, "answer": "4"}],
        args=types.SimpleNamespace(pass_at_k=False),
    )

    assert fitnesses == [1.0]
    assert "ASSISTANT: QED" in logs[0]
