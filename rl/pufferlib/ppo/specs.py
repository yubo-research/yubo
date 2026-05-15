from __future__ import annotations

import numpy as np
import torch.nn as nn

from rl import backbone
from rl.core import continuous_actions

from .specs_combined_model import (  # noqa: F401
    _ActorCritic,
    _UpdateStats,
)
from .specs_dataclasses_basic import (  # noqa: F401
    _ActionSpec,
    _ObservationSpec,
    _TrainPlan,
)
from .specs_dataclasses_buffer import (  # noqa: F401
    _FlatBatch,
    _RolloutBuffer,
    _RuntimeState,
)


def normalize_action_bounds(low: np.ndarray, high: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    return continuous_actions.normalize_action_bounds(low, high, dim)


def init_linear(module: nn.Module, gain: float) -> None:
    backbone.init_linear_layers(module, gain=float(gain))
