from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward."""

    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))
