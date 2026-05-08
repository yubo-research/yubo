from __future__ import annotations

import dataclasses

import torch.nn as nn


@dataclasses.dataclass
class _QBundle:
    q1_backbone: nn.Module
    q1_head: nn.Module
    q2_backbone: nn.Module
    q2_head: nn.Module
    q1: nn.Module
    q2: nn.Module
