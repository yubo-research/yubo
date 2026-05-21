from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import tomllib

MAMBA_ROOT_PREFIX = "/opt/conda"
HYPERSCALEES_ENV_NAME = "yubo-hyperscalees"
ISAACLAB_ENV_NAME = "yubo-isaaclab"
ISAACLAB_ENV_PREFIX = "isaaclab:"


def env_prefix(env_name: str) -> Path:
    return Path(MAMBA_ROOT_PREFIX) / "envs" / env_name


def local_experiment_config_arg(args: Sequence[str]) -> str | None:
    """Return the TOML path from `ops/experiment.py local <config>` args."""
    args = list(args)
    try:
        start = args.index("local") + 1
    except ValueError:
        return None

    skip_next = False
    for arg in args[start:]:
        if skip_next:
            skip_next = False
            continue
        if arg in {"-o", "--opt"}:
            skip_next = True
            continue
        if arg.startswith("-"):
            continue
        return arg
    return None


def experiment_env_tag(config_toml: str | Path) -> str | None:
    with Path(config_toml).open("rb") as f:
        data = tomllib.load(f)
    section = data.get("experiment")
    if section is None:
        section = data
    if not isinstance(section, dict):
        return None
    value = section.get("env_tag")
    return None if value is None else str(value)


def target_env_for_config(config_toml: str | Path) -> str | None:
    env_tag = experiment_env_tag(config_toml)
    if env_tag is not None and env_tag.strip().startswith(ISAACLAB_ENV_PREFIX):
        return ISAACLAB_ENV_NAME
    return None


def current_micromamba_env(environ: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if environ is None else environ
    name = env.get("CONDA_DEFAULT_ENV")
    if name:
        return Path(name).name
    prefix = env.get("CONDA_PREFIX")
    if prefix:
        return Path(prefix).name
    return None


def filtered_ld_library_path(target_env: str, existing: str | None = None) -> str:
    """Prefer the target env lib dir and remove stale micromamba env lib dirs."""
    target_lib = str(env_prefix(target_env) / "lib")
    parts = [target_lib]
    for part in (existing or "").split(":"):
        if not part or part == target_lib:
            continue
        if part.startswith(f"{MAMBA_ROOT_PREFIX}/envs/") and part.endswith("/lib"):
            continue
        parts.append(part)
    return ":".join(parts)


def reexec_command(
    *,
    target_env: str,
    script_path: str | Path,
    args: Sequence[str],
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    env = os.environ if environ is None else environ
    ld_library_path = filtered_ld_library_path(target_env, env.get("LD_LIBRARY_PATH"))
    return [
        "micromamba",
        "run",
        "-n",
        target_env,
        "env",
        f"LD_LIBRARY_PATH={ld_library_path}",
        "python",
        str(script_path),
        *list(args),
    ]


def reexec_environ(target_env: str, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if environ is None else environ)
    env["LD_LIBRARY_PATH"] = filtered_ld_library_path(target_env, env.get("LD_LIBRARY_PATH"))
    return env


def maybe_reexec_for_experiment_args(args: Sequence[str], *, script_path: str | Path) -> bool:
    config_toml = local_experiment_config_arg(args)
    if config_toml is None:
        return False
    target_env = target_env_for_config(config_toml)
    if target_env is None or current_micromamba_env() == target_env:
        return False
    if not env_prefix(target_env).exists():
        return False
    if shutil.which("micromamba") is None:
        return False

    cmd = reexec_command(target_env=target_env, script_path=script_path, args=args)
    print(f"[modal-runtime] routing config={config_toml!r} to micromamba env {target_env!r}", flush=True)
    os.execvpe(cmd[0], cmd, reexec_environ(target_env))
    return True
