from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from rl.torchrl.offpolicy import models as offpolicy_models

_ActorNet = offpolicy_models.ActorNet
_QNet = offpolicy_models.QNet
_QNetPixel = offpolicy_models.QNetPixel


class _ScaleActionToEnv(nn.Module):
    def __init__(self, action_low: np.ndarray, action_high: np.ndarray):
        super().__init__()
        self.register_buffer("_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("_high", torch.as_tensor(action_high, dtype=torch.float32))

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        width = (self._high - self._low).clamp(min=1e-08)
        return self._low + width * (1.0 + action) / 2.0
