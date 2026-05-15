import base64
import json
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

verifiers_types_mod = types.ModuleType("verifiers.types")
verifiers_types_mod.AssistantMessage = MagicMock
verifiers_types_mod.ToolCall = MagicMock
verifiers_types_mod.State = MagicMock
sys.modules.setdefault("verifiers", types.ModuleType("verifiers"))
sys.modules.setdefault("verifiers.types", verifiers_types_mod)

from llm.episode_proof import ProofEpisode  # noqa: E402
from llm.episodes import Case, RuntimeConfig  # noqa: E402
from llm.thm_task import (  # noqa: E402
    LanguageConfig,
    TheoremProvingTask,
    prime_sandbox_auth_configured,
    prime_sandbox_auth_summary,
)
from llm.thm_verifiers_env import TheoremVerifierEnv  # noqa: E402
from tests.thm_test_helpers import lean_proof_file_executor  # noqa: E402

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_thm_task_import_does_not_require_verifiers():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import llm.thm_task; print('ok')"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_prime_sandbox_auth_detects_env_key(tmp_path):
    assert prime_sandbox_auth_configured(environ={"PRIME_API_KEY": "test-key"}, home=tmp_path)


def test_prime_sandbox_auth_detects_prime_config(tmp_path):
    config = tmp_path / ".prime" / "config.json"
    config.parent.mkdir()
    config.write_text("{}", encoding="utf-8")

    assert prime_sandbox_auth_configured(environ={}, home=tmp_path)


def test_prime_sandbox_auth_missing_without_key_or_config(tmp_path):
    assert not prime_sandbox_auth_configured(environ={}, home=tmp_path)


def test_prime_sandbox_auth_summary_redacts_key(monkeypatch):
    monkeypatch.setenv("PRIME_API_KEY", "secret-prime-key")

    summary = prime_sandbox_auth_summary()

    assert "PRIME_API_KEY set" in summary
    assert "len=16" in summary
    assert "secret-prime-key" not in summary


class _ProofSandbox:
    def __init__(self, *, raise_on_write: bool = False):
        self.raise_on_write = raise_on_write
        self.content = ""
        self.compile_calls = 0
        self.deleted = []

    async def create(self, req):
        return types.SimpleNamespace(id="sandbox-id")

    async def wait_for_creation(self, sandbox_id):
        return None

    async def delete(self, sandbox_id):
        self.deleted.append(sandbox_id)

    async def execute_command(self, sandbox_id, command, timeout=None):
        if command.startswith("echo ") and "base64 -d >" in command:
            if self.raise_on_write:
                raise RuntimeError("write failed")
            self.content = base64.b64decode(command.split("echo ", 1)[1].split(" | base64 -d", 1)[0]).decode()
            return MagicMock(stdout="", stderr="", exit_code=0)
        if command.startswith("mkdir -p "):
            return MagicMock(stdout="", stderr="", exit_code=0)
        if command == "cat /workspace/tmp/proof.lean":
            return MagicMock(stdout=self.content, stderr="", exit_code=0)
        if "lean /workspace/tmp/proof.lean" in command:
            self.compile_calls += 1
            ok = "trivial" in self.content or "by rfl" in self.content
            return MagicMock(stdout="", stderr="" if ok else "error: failed", exit_code=0 if ok else 1)
        return MagicMock(stdout="", stderr="", exit_code=0)


class _ProofClient:
    def __init__(self, responses):
        self.responses = list(responses)

    async def create(self, messages):
        content = self.responses.pop(0) if self.responses else "theorem test : True := by\n  trivial"
        return MagicMock(choices=[MagicMock(message=types.SimpleNamespace(content=content, tool_calls=None))])


def _proof_episode(responses, *, max_turns=2, sandbox=None):
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
    env = TheoremVerifierEnv(lang_cfg, max_turns=max_turns)
    sandbox = sandbox or _ProofSandbox()

    def client_factory(*args, **kwargs):
        return _ProofClient(responses)

    return ProofEpisode(env, sandbox, client_factory=client_factory), sandbox


def _proof_case():
    return Case(
        id="lean:test",
        prompt="Prove it",
        target={"statement": "theorem test : True := by\n  sorry", "row": {}},
    )


@pytest.mark.anyio
async def test_proof_episode_verified_first_turn():
    episode, sandbox = _proof_episode(["theorem test : True := by\n  trivial"])

    signal = await episode.run(_proof_case(), MagicMock(), {}, RuntimeConfig())

    assert signal.reward == 1.0
    assert signal.status == "ok"
    assert signal.metrics["compile_calls"] == 1
    assert [turn.kind for turn in signal.turns] == ["system", "user", "model", "tool", "system"]
    assert sandbox.deleted == ["sandbox-id"]


@pytest.mark.anyio
async def test_proof_episode_uses_compiler_feedback_for_retry():
    episode, _sandbox = _proof_episode(
        [
            "theorem test : True := by\n  exact False.elim",
            "theorem test : True := by\n  trivial",
        ],
        max_turns=2,
    )

    signal = await episode.run(_proof_case(), MagicMock(), {}, RuntimeConfig())

    assert signal.reward == 1.0
    assert signal.status == "ok"
    assert signal.metrics["compile_calls"] == 2
    assert sum(1 for turn in signal.turns if turn.kind == "tool") == 2


@pytest.mark.anyio
async def test_proof_episode_rejects_placeholder():
    episode, _sandbox = _proof_episode(["theorem test : True := by\n  sorry"], max_turns=1)

    signal = await episode.run(_proof_case(), MagicMock(), {}, RuntimeConfig())

    assert signal.reward == 0.0
    assert signal.status == "placeholder"
    assert signal.metrics["placeholder"] is True


@pytest.mark.anyio
async def test_proof_episode_marks_tool_error():
    episode, _sandbox = _proof_episode(
        ["theorem test : True := by\n  trivial"],
        sandbox=_ProofSandbox(raise_on_write=True),
        max_turns=1,
    )

    signal = await episode.run(_proof_case(), MagicMock(), {}, RuntimeConfig())

    assert signal.reward == 0.0
    assert signal.status == "tool_error"
    assert signal.metrics["tool_error"] is True


@pytest.mark.anyio
async def test_proof_episode_marks_wrong_after_max_turns():
    episode, _sandbox = _proof_episode(
        [
            "theorem test : True := by\n  exact False.elim",
            "theorem test : True := by\n  exact False.elim",
        ],
        max_turns=2,
    )

    signal = await episode.run(_proof_case(), MagicMock(), {}, RuntimeConfig())

    assert signal.reward == 0.0
    assert signal.status == "wrong"
    assert signal.metrics["compile_calls"] == 2


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


@pytest.mark.anyio
async def test_thm_task_creates_proof_parent_directory(monkeypatch):
    verifiers_types_mod = types.ModuleType("verifiers.types")
    verifiers_types_mod.AssistantMessage = MagicMock
    verifiers_types_mod.ToolCall = MagicMock
    monkeypatch.setitem(sys.modules, "verifiers", types.ModuleType("verifiers"))
    monkeypatch.setitem(sys.modules, "verifiers.types", verifiers_types_mod)

    prime_sandboxes_mod = types.ModuleType("prime_sandboxes")
    prime_sandboxes_mod.CreateSandboxRequest = MagicMock
    prime_sandboxes_mod.AsyncSandboxClient = MagicMock
    monkeypatch.setitem(sys.modules, "prime_sandboxes", prime_sandboxes_mod)

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
    task.lang_cfg = lang_cfg

    answer = {"statement": "theorem test : 1 + 1 = 2 := rfl", "row": {}}

    sandbox_client = AsyncMock()
    sandbox = MagicMock()
    sandbox.id = "test-sandbox-id"
    sandbox_client.create.return_value = sandbox

    from verifiers.types import AssistantMessage

    mock_client = AsyncMock()
    mock_client.create.side_effect = [
        MagicMock(choices=[MagicMock(message=AssistantMessage(content="QED"))]),
    ]
    monkeypatch.setattr("llm.thm_task._VLLMRLMClient", MagicMock(return_value=mock_client))

    sandbox_client.execute_command.return_value = MagicMock(stdout="", stderr="", exit_code=0)

    await task._setup_initial_proof("test-sandbox-id", answer, sandbox_client)

    assert sandbox_client.execute_command.call_args_list[0].args[1].startswith("mkdir -p ")
    assert sandbox_client.execute_command.call_args_list[1].args[1].startswith("echo ")
    b64_content = sandbox_client.execute_command.call_args_list[1].args[1].split("echo ", 1)[1].split(" | base64 -d", 1)[0]
    content = base64.b64decode(b64_content).decode()
    assert "import Mathlib" in content
    assert "import Aesop" in content
    assert "set_option maxHeartbeats 0" in content
    assert "open BigOperators Real Nat Topology Rat" in content
    assert "theorem test : 1 + 1 = 2 := rfl" in content


@pytest.mark.anyio
async def test_lean_env_executes_raw_model_code_as_synthetic_tool(monkeypatch):
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

    from verifiers.types import AssistantMessage

    client = AsyncMock()
    client.create.return_value = MagicMock(choices=[MagicMock(message=AssistantMessage(content="theorem test : 1 + 1 = 2 := by rfl", tool_calls=None))])
    sandbox_client = AsyncMock()
    proof_file = {"content": ""}

    sandbox_client.execute_command.side_effect = lean_proof_file_executor(proof_file)

    sandbox_client.create.return_value = types.SimpleNamespace(id="test-sandbox-id")
    signal = await ProofEpisode(
        env,
        sandbox_client,
        client_factory=lambda *args, **kwargs: client,
    ).run(
        Case(
            id="lean:test",
            prompt="Prove it",
            target={"statement": "theorem test : 1 + 1 = 2 := sorry"},
        ),
        MagicMock(),
        {},
        RuntimeConfig(),
    )

    commands = [call.args[1] for call in sandbox_client.execute_command.call_args_list]
    write_cmds = [cmd for cmd in commands if "base64 -d > '/workspace/tmp/proof.lean'" in cmd]
    assert write_cmds
    written = proof_file["content"]
    assert "import Mathlib" in written
    assert "set_option maxHeartbeats 0" in written
    assert "theorem test : 1 + 1 = 2 := by rfl" in written
    assert signal.reward == 1.0
    assert any(turn.kind == "tool" and turn.name == "lean4" for turn in signal.turns)


@pytest.mark.anyio
async def test_lean_rubric_rejects_compiling_placeholder(monkeypatch):
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

    from verifiers.types import AssistantMessage

    client = AsyncMock()
    client.create.return_value = MagicMock(choices=[MagicMock(message=AssistantMessage(content="theorem test : True := by\n  sorry", tool_calls=None))])
    sandbox_client = AsyncMock()
    proof_file = {"content": ""}

    sandbox_client.execute_command.side_effect = lean_proof_file_executor(proof_file)

    sandbox_client.create.return_value = types.SimpleNamespace(id="test-sandbox-id")
    signal = await ProofEpisode(
        env,
        sandbox_client,
        client_factory=lambda *args, **kwargs: client,
    ).run(
        Case(
            id="lean:test",
            prompt="Prove it",
            target={"statement": "theorem test : True := by\n  sorry"},
        ),
        MagicMock(),
        {},
        RuntimeConfig(),
    )

    assert signal.reward == 0.0
