#!/usr/bin/env python3
"""Verify Pixi env imports (no install side effects)."""

from __future__ import annotations

import os
import sys

import click

from admin.check_env_common import check_core, check_numpy_numba


def _default_jax_platform() -> str:
    return os.environ.get("JAX_PLATFORMS", "mps" if sys.platform == "darwin" else "cpu")


def _hint_missing(name: str) -> str:
    return f"Missing {name}. {_pixi_task_hint('setup')}"


def _pixi_task_hint(task: str) -> str:
    env_name = os.environ.get("PIXI_ENVIRONMENT_NAME")
    env_arg = f"-e {env_name} " if env_name else "-e <env> "
    return f"Run: pixi run {env_arg}{task}"


def _check_mac(*, jax_platform: str, require_llm: bool) -> None:
    import botorch  # noqa: F401
    import gpytorch  # noqa: F401
    import jax
    import jax_plugins.mps  # noqa: F401
    import torch

    if not torch.backends.mps.is_available():
        raise SystemExit("torch MPS is not available on this Mac")
    devices = list(map(str, jax.devices()))
    print("mac jax", jax.__version__, "devices", devices, "jax_platform", jax_platform)
    if require_llm:
        try:
            import vllm
            import vllm_metal  # noqa: F401
        except ImportError as exc:
            raise SystemExit(f"{exc}\n{_pixi_task_hint('llm-mac')}") from exc

        print("mac llm ok", "vllm", vllm.__version__)


def _check_linux() -> None:
    import vllm  # noqa: F401

    print("linux ok", "vllm importable")


@click.command(help=__doc__)
@click.option(
    "--require-llm",
    is_flag=True,
    help="Also require vllm + vllm_metal (Mac LLM lane).",
)
@click.option(
    "--jax-platform",
    metavar="NAME",
    default=None,
    help="JAX backend to assert. Defaults to JAX_PLATFORMS, then mps on darwin or cpu elsewhere.",
)
def main(*, require_llm: bool, jax_platform: str | None) -> None:
    resolved_jax_platform = jax_platform if jax_platform is not None else _default_jax_platform()
    os.environ.setdefault("JAX_PLATFORMS", resolved_jax_platform)
    check_numpy_numba()
    check_core(hint=_hint_missing("BO extras"))
    if sys.platform == "darwin":
        _check_mac(jax_platform=resolved_jax_platform, require_llm=require_llm)
    else:
        _check_linux()
    print("pixi env check passed")


if __name__ == "__main__":
    main()
