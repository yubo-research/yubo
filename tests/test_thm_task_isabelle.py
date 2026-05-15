import base64
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm.thm_task import LanguageConfig, TheoremProvingTask


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_isabelle_backend_writes_session_root_and_proof(monkeypatch):
    verifiers_types_mod = types.ModuleType("verifiers.types")
    verifiers_types_mod.AssistantMessage = MagicMock
    verifiers_types_mod.ToolCall = MagicMock
    verifiers_types_mod.State = MagicMock
    monkeypatch.setitem(sys.modules, "verifiers", types.ModuleType("verifiers"))
    monkeypatch.setitem(sys.modules, "verifiers.types", verifiers_types_mod)

    prime_sandboxes_mod = types.ModuleType("prime_sandboxes")
    prime_sandboxes_mod.CreateSandboxRequest = MagicMock
    prime_sandboxes_mod.AsyncSandboxClient = MagicMock
    monkeypatch.setitem(sys.modules, "prime_sandboxes", prime_sandboxes_mod)

    lang_cfg = LanguageConfig(
        name="isabelle",
        extension="thy",
        docker_image="mock-image",
        compile_cmd="isabelle build -D {session_dir}",
        guard_begin="(* guard begin *)",
        guard_end="(* guard end *)",
        workdir="/home/isabelle",
        proof_path="/yubo_proof/Proof",
        session_dir="/home/isabelle/yubo_proof",
        root_file="/home/isabelle/yubo_proof/ROOT",
    )

    task = TheoremProvingTask(batch_size=1, language="lean4")
    task.lang_cfg = lang_cfg
    sandbox_client = AsyncMock()
    sandbox_client.execute_command.return_value = MagicMock(stdout="", stderr="", exit_code=0)

    await task._setup_initial_proof(
        "test-sandbox-id",
        {"statement": "theorem test : True := by trivial", "row": {}},
        sandbox_client,
    )

    commands = [call.args[1] for call in sandbox_client.execute_command.call_args_list]
    assert any(cmd.startswith("mkdir -p '/home/isabelle/yubo_proof'") for cmd in commands)
    assert any(cmd.startswith("echo ") and "ROOT" not in cmd for cmd in commands)
    root_cmds = [cmd for cmd in commands if cmd.startswith("echo ") and "base64 -d > '/home/isabelle/yubo_proof/ROOT'" in cmd]
    assert len(root_cmds) == 1
    b64_root = root_cmds[0].split("echo ", 1)[1].split(" | base64 -d", 1)[0]
    assert "session YuboProof = Pure" in base64.b64decode(b64_root).decode()
