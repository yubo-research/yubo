import sys
import types
from unittest.mock import MagicMock

import pytest

from llm.tasks_verifiers import _ENV_CACHE, VerifiersTask
from llm.tasks_verifiers_utils import require_verifiers_runtime
from tests.llm_test_mocks_env import FakeEnv
from tests.llm_test_mocks_state import FakeAssistantMessage, FakeState
from tests.llm_test_mocks_vllm import FakeRLM


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_verifiers_format_prompt_uses_chat_template_for_message_lists():
    from llm.tasks_verifiers_utils import format_prompt

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            assert tokenize is False
            assert add_generation_prompt is True
            assert messages == [
                {"role": "system", "content": "Use boxes."},
                {"role": "user", "content": "2+2?"},
            ]
            return "<chat>system:Use boxes.|user:2+2?|assistant:"

    prompt = format_prompt(
        [
            {"role": "system", "content": "Use boxes."},
            {"role": "user", "content": "2+2?"},
        ],
        tokenizer=FakeTokenizer(),
        apply_chat_template=True,
    )

    assert prompt == "<chat>system:Use boxes.|user:2+2?|assistant:"


def test_verifiers_turn_logging_keeps_model_response():
    from llm.episode_verifiers import _turns_from_state
    from llm.episodes import Case, Signal, signal_log

    dict_state = FakeState(
        {
            "trajectory": [
                {"role": "assistant", "content": "The answer is \\boxed{4}."},
            ],
        }
    )
    dict_turns = _turns_from_state(dict_state)
    assert dict_turns[0].kind == "assistant"
    assert dict_turns[0].text == "The answer is \\boxed{4}."

    step_state = FakeState(
        {
            "trajectory": [
                types.SimpleNamespace(completion=[FakeAssistantMessage("TrajectoryStep response \\boxed{4}.")]),
            ],
        }
    )
    step_turns = _turns_from_state(step_state)
    assert step_turns[0].kind == "assistant"
    assert step_turns[0].text == "TrajectoryStep response \\boxed{4}."

    fallback_state = FakeState(
        {
            "trajectory": [{"role": "unknown"}],
            "completion": [FakeAssistantMessage("Fallback response \\boxed{4}.")],
        }
    )
    fallback_turns = _turns_from_state(fallback_state)
    log = signal_log(
        Case(id="case-0", prompt="2+2?", target="4"),
        Signal(reward=1.0, status="ok", turns=tuple(fallback_turns)),
    )

    assert "ASSISTANT: Fallback response \\boxed{4}." in log


def test_verifiers_runtime_preflight_reports_openai_agents_mismatch(monkeypatch):
    import llm.tasks_verifiers_utils as utils

    versions = {
        "verifiers": "0.1.15.dev2",
        "openai-agents": "0.17.2",
        "openai": "2.24.0",
    }

    def version(name):
        if name not in versions:
            raise utils.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(utils.metadata, "version", version)
    monkeypatch.setattr(
        utils.metadata,
        "requires",
        lambda name: ["openai<3,>=2.26.0"] if name == "openai-agents" else [],
    )

    with pytest.raises(RuntimeError, match="openai==2.24.0"):
        require_verifiers_runtime()


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
