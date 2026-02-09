from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


@dataclass(frozen=True)
class CheckpointManager:
    exp_dir: Path
    checkpoints_dirname: str = "checkpoints"
    latest_filename: str = "checkpoint_last.pt"

    def paths(self, iteration: int) -> tuple[Path, Path]:
        ckpt_dir = self.exp_dir / self.checkpoints_dirname
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"iter_{int(iteration):06d}.pt"
        latest_path = self.exp_dir / self.latest_filename
        return ckpt_path, latest_path

    def save_both(self, state: dict[str, Any], *, iteration: int) -> tuple[Path, Path]:
        ckpt_path, latest_path = self.paths(iteration)
        save_checkpoint(ckpt_path, state)
        save_checkpoint(latest_path, state)
        return ckpt_path, latest_path
