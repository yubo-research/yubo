from __future__ import annotations

from pathlib import Path

import torch

from rl.algos.checkpointing import (
    CheckpointManager,
    append_jsonl,
    load_checkpoint,
    save_checkpoint,
)


def _checkpoint_paths(exp_dir: Path, iteration: int) -> tuple[Path, Path]:
    return CheckpointManager(exp_dir=exp_dir).paths(iteration)


def _save_checkpoint(path: Path, state: dict) -> None:
    save_checkpoint(path, state)


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    return load_checkpoint(path, device)


def _append_jsonl(path: Path, record: dict) -> None:
    append_jsonl(path, record)
