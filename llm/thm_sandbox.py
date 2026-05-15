from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


@dataclass(frozen=True)
class SandboxCreateRequest:
    docker_image: str
    name: str


@dataclass(frozen=True)
class SandboxCommandResult:
    stdout: str
    stderr: str
    exit_code: int


class LocalDockerSandboxClient:
    """Prime-compatible async sandbox client backed by local Docker or Podman."""

    def __init__(
        self,
        *,
        runtime: str | None = None,
        network: str = "none",
        memory: str | None = None,
        cpus: str | None = None,
        pids_limit: int | None = 512,
        entrypoint: str | None = None,
        user: str | None = None,
        path_prefix: str | None = None,
    ) -> None:
        self.runtime = runtime or os.environ.get("THM_DOCKER_RUNTIME", "docker")
        self.runtime_cmd = shlex.split(self.runtime)
        self.network = network
        self.memory = memory or os.environ.get("THM_DOCKER_MEMORY", "4g")
        self.cpus = cpus or os.environ.get("THM_DOCKER_CPUS", "2")
        self.pids_limit = pids_limit
        self.entrypoint = entrypoint or os.environ.get("THM_DOCKER_ENTRYPOINT", "/bin/sh")
        self.user = user if user is not None else os.environ.get("THM_DOCKER_USER", "0:0")
        self.path_prefix = path_prefix if path_prefix is not None else os.environ.get("THM_DOCKER_PATH_PREFIX", "/home/lean/.elan/bin")
        self.home = os.environ.get("THM_DOCKER_HOME", "/home/lean")

    async def create(self, req: SandboxCreateRequest) -> Any:
        name = _container_name(req.name)
        cmd = [
            *self.runtime_cmd,
            "run",
            "-d",
            "--name",
            name,
            "--network",
            self.network,
            "--security-opt",
            "no-new-privileges",
            "--entrypoint",
            self.entrypoint,
        ]
        if self.user:
            cmd.extend(["--user", self.user])
        if self.memory:
            cmd.extend(["--memory", self.memory])
        if self.cpus:
            cmd.extend(["--cpus", self.cpus])
        if self.pids_limit is not None:
            cmd.extend(["--pids-limit", str(int(self.pids_limit))])
        cmd.extend([req.docker_image, "-lc", "sleep infinity"])

        result = await _run_subprocess(cmd, timeout=120)
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to create local theorem sandbox with {self.runtime}: {result.stderr.strip() or result.stdout.strip()}")
        container_id = (result.stdout.strip().splitlines() or [name])[0]
        return SimpleNamespace(id=container_id, name=name)

    async def wait_for_creation(self, sandbox_id: str) -> None:
        result = await _run_subprocess([*self.runtime_cmd, "inspect", sandbox_id], timeout=30)
        if result.exit_code != 0:
            raise RuntimeError(f"Local theorem sandbox was not created: {result.stderr.strip()}")

    async def execute_command(self, sandbox_id: str, command: str, timeout: int | float | None = None) -> SandboxCommandResult:
        if self.home:
            command = f"export HOME={_shell_quote(self.home)}; {command}"
        if self.path_prefix:
            command = f"export PATH={_shell_quote(self.path_prefix)}:$PATH; {command}"
        return await _run_subprocess(
            [*self.runtime_cmd, "exec", sandbox_id, self.entrypoint, "-lc", command],
            timeout=timeout,
        )

    async def delete(self, sandbox_id: str) -> None:
        await _run_subprocess([*self.runtime_cmd, "rm", "-f", sandbox_id], timeout=30)


async def build_sandbox_client() -> Any:
    backend = selected_sandbox_backend()

    if backend in {"docker", "local_docker", "local-docker"}:
        return LocalDockerSandboxClient()
    if backend == "podman":
        return LocalDockerSandboxClient(runtime="podman")
    if backend == "prime":
        require_prime_sandbox_auth()
        from prime_sandboxes import AsyncSandboxClient

        return AsyncSandboxClient()
    raise ValueError(f"Unsupported THM_SANDBOX_BACKEND={backend!r}. Use prime, local_docker, docker, or podman.")


def build_sandbox_create_request(docker_image: str, name: str) -> Any:
    backend = selected_sandbox_backend()
    if backend == "prime":
        from prime_sandboxes import CreateSandboxRequest

        return CreateSandboxRequest(docker_image=docker_image, name=name)
    return SandboxCreateRequest(docker_image=docker_image, name=name)


def selected_sandbox_backend() -> str:
    backend = os.environ.get("THM_SANDBOX_BACKEND", "").strip().lower()
    return backend or "local_docker"


def prime_sandbox_auth_configured() -> bool:
    if os.environ.get("PRIME_API_KEY"):
        return True
    return (Path.home() / ".prime" / "config.json").is_file()


def require_prime_sandbox_auth() -> None:
    if prime_sandbox_auth_configured():
        return
    raise RuntimeError(
        "Theorem proving is configured for Prime Sandboxes, but no Prime auth is configured. Set PRIME_API_KEY or use THM_SANDBOX_BACKEND=local_docker."
    )


def require_sandbox_backend_ready(language: str | None = None) -> None:
    backend = selected_sandbox_backend()
    if backend == "prime":
        require_prime_sandbox_auth()
        return
    if backend in {"docker", "local_docker", "local-docker"}:
        runtime = os.environ.get("THM_DOCKER_RUNTIME", "docker")
    elif backend == "podman":
        runtime = "podman"
    else:
        return
    runtime_exe = shlex.split(runtime)[0] if runtime else ""
    if shutil.which(runtime_exe) is None:
        raise RuntimeError(
            f"Theorem proving is configured for {backend}, but {runtime_exe!r} is not on PATH. Install Docker/Podman or set THM_SANDBOX_BACKEND=prime."
        )
    _require_runtime_usable(runtime, backend)
    lang = _language_name(language)
    image = _image_for_language(lang)
    if image:
        _require_image_present(runtime, image)
    if lang == "lean4":
        _require_lean4_image_offline_ready(runtime, image)


def is_prime_unauthorized_error(exc: BaseException) -> bool:
    cls = type(exc)
    return cls.__name__ == "UnauthorizedError" and cls.__module__.startswith("prime_sandboxes")


async def _run_subprocess(cmd: list[str], *, timeout: int | float | None) -> SandboxCommandResult:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        stdout_b, stderr_b = await proc.communicate()
        return SandboxCommandResult(
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=(stderr_b.decode("utf-8", errors="replace") + f"\nCommand timed out after {timeout}s").strip(),
            exit_code=124,
        )
    return SandboxCommandResult(
        stdout=stdout_b.decode("utf-8", errors="replace"),
        stderr=stderr_b.decode("utf-8", errors="replace"),
        exit_code=int(proc.returncode or 0),
    )


def _require_runtime_usable(runtime: str, backend: str) -> None:
    result = _run_checked([*shlex.split(runtime), "version"], timeout=20)
    if result.returncode != 0:
        raise RuntimeError(f"Theorem proving is configured for {backend}, but {runtime!r} is not usable: {_subprocess_output(result)}")


def _require_image_present(runtime: str, image: str) -> None:
    result = _run_checked([*shlex.split(runtime), "image", "inspect", image], timeout=30)
    if result.returncode != 0:
        raise RuntimeError(
            f"Theorem proving image {image!r} is not available locally. Build it first with `./ops/build_thm_image.sh lean4`. {_subprocess_output(result)}"
        )


def _require_lean4_image_offline_ready(runtime: str, image: str) -> None:
    if not image:
        return
    code = "import Mathlib\n\nexample : 1 + 1 = 2 := by norm_num\n"
    command = (
        "export HOME='/home/lean'; "
        "export PATH='/home/lean/.elan/bin':$PATH; "
        "cd /workspace; "
        f"printf %s {shlex.quote(code)} > /tmp/yubo_preflight.lean; "
        "yubo-lean /tmp/yubo_preflight.lean"
    )
    result = _run_checked(
        [
            *shlex.split(runtime),
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "/bin/sh",
            image,
            "-lc",
            command,
        ],
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Lean4 theorem image {image!r} is not offline-ready. Rebuild it with `./ops/build_thm_image.sh lean4`. {_subprocess_output(result)}"
        )


def _run_checked(cmd: list[str], *, timeout: int | float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(cmd, 124, stdout=_process_text(exc.stdout), stderr=f"timed out after {timeout}s")


def _subprocess_output(result: subprocess.CompletedProcess[str]) -> str:
    output = ((result.stderr or "") + "\n" + (result.stdout or "")).strip()
    return output or f"exit_code={result.returncode}"


def _process_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _language_name(language: str | None) -> str | None:
    if language is None:
        return None
    text = str(language)
    if text.startswith("thm:"):
        parts = text.split(":")
        return parts[1] if len(parts) > 1 else None
    return text


def _image_for_language(language: str | None) -> str | None:
    if language == "lean4":
        return os.environ.get("THM_LEAN4_DOCKER_IMAGE", "yubo-lean4-mathlib:latest")
    if language == "coq":
        return os.environ.get("THM_COQ_DOCKER_IMAGE", "coqorg/coq:8.19")
    if language == "isabelle":
        return os.environ.get("THM_ISABELLE_DOCKER_IMAGE", "makarius/isabelle:latest")
    return None


def _container_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "_.-" else "-" for ch in name.lower())
    return cleaned[:120].strip("-") or "yubo-thm"


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


__all__ = [
    "LocalDockerSandboxClient",
    "SandboxCommandResult",
    "SandboxCreateRequest",
    "build_sandbox_client",
    "build_sandbox_create_request",
    "is_prime_unauthorized_error",
    "prime_sandbox_auth_configured",
    "require_prime_sandbox_auth",
    "require_sandbox_backend_ready",
    "selected_sandbox_backend",
]
