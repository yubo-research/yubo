from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from llm import thm_sandbox

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_selected_sandbox_backend_defaults_to_local_docker(monkeypatch):
    monkeypatch.delenv("THM_SANDBOX_BACKEND", raising=False)

    assert thm_sandbox.selected_sandbox_backend() == "local_docker"


def test_build_sandbox_create_request_uses_local_request_by_default(monkeypatch):
    monkeypatch.delenv("THM_SANDBOX_BACKEND", raising=False)

    req = thm_sandbox.build_sandbox_create_request("image", "name")

    assert req == thm_sandbox.SandboxCreateRequest(docker_image="image", name="name")


def test_require_sandbox_backend_ready_checks_docker_binary(monkeypatch):
    monkeypatch.setenv("THM_SANDBOX_BACKEND", "local_docker")
    monkeypatch.setattr(thm_sandbox.shutil, "which", lambda runtime: None)

    with pytest.raises(RuntimeError, match="not on PATH"):
        thm_sandbox.require_sandbox_backend_ready()


def test_require_sandbox_backend_ready_checks_lean_image_offline(monkeypatch):
    monkeypatch.setenv("THM_SANDBOX_BACKEND", "local_docker")
    monkeypatch.setenv("THM_LEAN4_DOCKER_IMAGE", "lean-img")
    monkeypatch.setattr(thm_sandbox.shutil, "which", lambda runtime: "/usr/bin/docker")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(thm_sandbox.subprocess, "run", fake_run)

    thm_sandbox.require_sandbox_backend_ready("thm:lean4:dataset")

    assert calls[0] == ["docker", "version"]
    assert calls[1] == ["docker", "image", "inspect", "lean-img"]
    assert calls[2][:7] == [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--entrypoint",
        "/bin/sh",
    ]
    assert calls[2][7] == "lean-img"
    assert "yubo-lean /tmp/yubo_preflight.lean" in calls[2][-1]


def test_require_sandbox_backend_ready_reports_offline_preflight_failure(monkeypatch):
    monkeypatch.setenv("THM_SANDBOX_BACKEND", "local_docker")
    monkeypatch.setattr(thm_sandbox.shutil, "which", lambda runtime: "/usr/bin/docker")

    def fake_run(cmd, **kwargs):
        if "run" in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="git clone failed")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(thm_sandbox.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="offline-ready"):
        thm_sandbox.require_sandbox_backend_ready("lean4")


async def test_local_docker_sandbox_client_create_exec_delete(monkeypatch):
    calls = []

    async def fake_run_subprocess(cmd, *, timeout):
        calls.append((cmd, timeout))
        if cmd[:2] == ["docker", "run"]:
            return thm_sandbox.SandboxCommandResult(stdout="container-id\n", stderr="", exit_code=0)
        return thm_sandbox.SandboxCommandResult(stdout="out", stderr="", exit_code=0)

    monkeypatch.setattr(thm_sandbox, "_run_subprocess", fake_run_subprocess)

    client = thm_sandbox.LocalDockerSandboxClient(runtime="docker", memory="1g", cpus="1")
    sandbox = await client.create(thm_sandbox.SandboxCreateRequest(docker_image="ubuntu:22.04", name="Test Box"))
    await client.wait_for_creation(sandbox.id)
    result = await client.execute_command(sandbox.id, "echo hi", timeout=3)
    await client.delete(sandbox.id)

    assert sandbox == SimpleNamespace(id="container-id", name="test-box")
    assert result.stdout == "out"
    assert calls[0][0][:4] == ["docker", "run", "-d", "--name"]
    assert "--network" in calls[0][0]
    assert "--entrypoint" in calls[0][0]
    assert "--user" in calls[0][0]
    assert "0:0" in calls[0][0]
    assert calls[0][0][-3:] == ["ubuntu:22.04", "-lc", "sleep infinity"]
    assert calls[2][0] == [
        "docker",
        "exec",
        "container-id",
        "/bin/sh",
        "-lc",
        "export PATH='/home/lean/.elan/bin':$PATH; export HOME='/home/lean'; echo hi",
    ]
    assert calls[3][0] == ["docker", "rm", "-f", "container-id"]


async def test_local_docker_sandbox_client_supports_sudo_runtime(monkeypatch):
    calls = []

    async def fake_run_subprocess(cmd, *, timeout):
        calls.append(cmd)
        return thm_sandbox.SandboxCommandResult(stdout="container-id\n", stderr="", exit_code=0)

    monkeypatch.setattr(thm_sandbox, "_run_subprocess", fake_run_subprocess)

    client = thm_sandbox.LocalDockerSandboxClient(runtime="sudo -n docker")
    sandbox = await client.create(thm_sandbox.SandboxCreateRequest(docker_image="ubuntu:22.04", name="box"))
    await client.delete(sandbox.id)

    assert calls[0][:3] == ["sudo", "-n", "docker"]
    assert calls[1][:3] == ["sudo", "-n", "docker"]


async def test_prime_backend_requires_auth_before_import(monkeypatch):
    monkeypatch.setenv("THM_SANDBOX_BACKEND", "prime")
    monkeypatch.delenv("PRIME_API_KEY", raising=False)
    monkeypatch.setattr(
        thm_sandbox.Path,
        "home",
        lambda *args: thm_sandbox.Path("/tmp/yubo-no-prime-config"),
    )

    with pytest.raises(RuntimeError, match="configured for Prime"):
        await thm_sandbox.build_sandbox_client()
