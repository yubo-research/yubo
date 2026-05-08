from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass(frozen=True)
class SACUpdateBatch:
    obs: torch.Tensor
    act: torch.Tensor
    rew: torch.Tensor
    nxt: torch.Tensor
    done: torch.Tensor
