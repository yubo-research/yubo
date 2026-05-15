import base64
from unittest.mock import MagicMock


def lean_proof_file_executor(proof_file: dict[str, str]):
    async def execute_command(_sandbox_id, command, timeout=None):
        if command.startswith("echo ") and "base64 -d > '/workspace/tmp/proof.lean'" in command:
            proof_file["content"] = base64.b64decode(command.split("echo ", 1)[1].split(" | base64 -d", 1)[0]).decode()
        if command == "cat /workspace/tmp/proof.lean":
            return MagicMock(stdout=proof_file["content"], stderr="", exit_code=0)
        return MagicMock(stdout="", stderr="", exit_code=0)

    return execute_command
