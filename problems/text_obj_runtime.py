from __future__ import annotations

import importlib.util
import os
import tempfile
from typing import Any


_RUNTIME_ERROR = (
    "Real UHD text runs require the CUDA text runtime: ray, vllm, transformers, "
    "torch, peft, accelerate, and safetensors. Run admin/setup-hyperscalees.sh on "
    "the CUDA machine, then launch with ./ops/exp_uhd.py from that environment."
)


def require_runtime() -> None:
    missing = [
        module
        for module in (
            "accelerate",
            "peft",
            "ray",
            "safetensors",
            "torch",
            "transformers",
            "vllm",
        )
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise RuntimeError(f"{_RUNTIME_ERROR} Missing: {', '.join(sorted(missing))}.")


def make_adapter_root() -> str:
    parent = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None
    return tempfile.mkdtemp(prefix="yubo_text_uhd_", dir=parent)


def base_seed(cfg: Any) -> int:
    seed = next(
        (candidate for candidate in (cfg.noise_seed_0, cfg.problem_seed, 0) if candidate is not None),
        0,
    )
    return int(seed) + int(cfg.seed_offset)
