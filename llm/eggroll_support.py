from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from llm.config import LLMConfig
from llm.runtime_messages import missing_runtime_message


def write_run_config(exp_dir: Path, cfg: LLMConfig) -> None:
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2, sort_keys=True)


def adapter_root_for(exp_dir: Path) -> str:
    digest = hashlib.sha1(str(exp_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    return str(Path("/dev/shm") / f"yubo_llm_lora_{digest}")


def base_seed(cfg: LLMConfig) -> int:
    seed = cfg.noise_seed_0
    if seed is None:
        seed = cfg.problem_seed
    if seed is None:
        seed = 0
    return int(seed) + int(cfg.seed_offset)


def eggroll_missing_runtime_message(missing: list[str]) -> str:
    return missing_runtime_message("EggRoll", missing, "./ops/llm.py")


__all__ = [
    "adapter_root_for",
    "base_seed",
    "eggroll_missing_runtime_message",
    "write_run_config",
]
