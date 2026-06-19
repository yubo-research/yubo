#!/usr/bin/env python3
"""Verify Pixi env imports (no install side effects)."""

from __future__ import annotations

import argparse
import os
import sys


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--require-llm",
        action="store_true",
        help="Also require vllm + vllm_metal (Mac LLM lane).",
    )
    p.add_argument(
        "--jax-platform",
        default=os.environ.get("JAX_PLATFORMS", "mps" if sys.platform == "darwin" else "cpu"),
        help="JAX backend to assert (default: mps on darwin, else cpu).",
    )
    return p.parse_args()


def _check_numpy_numba() -> None:
    import numba
    import numpy as np

    major, minor = (int(x) for x in np.__version__.split(".")[:2])
    if (major, minor) < (2, 3):
        raise SystemExit(f"numpy {np.__version__} too old; need >= 2.3")
    print("numpy", np.__version__, "numba", numba.__version__)


def _hint_missing(name: str) -> str:
    return f"Missing {name}. {_pixi_task_hint('setup')}"


def _pixi_task_hint(task: str) -> str:
    env_name = os.environ.get("PIXI_ENVIRONMENT_NAME")
    env_arg = f"-e {env_name} " if env_name else "-e <env> "
    return f"Run: pixi run {env_arg}{task}"


def _check_core() -> None:
    mac = sys.platform == "darwin"
    try:
        import faiss  # noqa: F401
        import hyperscalees  # noqa: F401
        from enn.enn.enn_class import EpistemicNearestNeighbors

        if not mac:
            import kinetix.environment  # noqa: F401
        import LassoBench  # noqa: F401
        import torch
        from pyvecch.input_transforms import Identity
    except ImportError as exc:
        raise SystemExit(f"{exc}\n{_hint_missing('BO extras')}") from exc

    print(
        "core ok",
        EpistemicNearestNeighbors.__name__,
        Identity.__name__,
        "torch",
        torch.__version__,
    )


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


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("JAX_PLATFORMS", args.jax_platform)
    _check_numpy_numba()
    _check_core()
    if sys.platform == "darwin":
        _check_mac(jax_platform=args.jax_platform, require_llm=args.require_llm)
    else:
        _check_linux()
    print("pixi env check passed")


if __name__ == "__main__":
    main()
