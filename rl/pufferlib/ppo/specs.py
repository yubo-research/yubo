from __future__ import annotations

import numpy as np
import torch.nn as nn

from rl.backbone import init_linear_layers as _init_linear_layers_shared
from rl.core.continuous_actions import (
    normalize_action_bounds as _normalize_action_bounds_shared,
)

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
    return _normalize_action_bounds_shared(low, high, dim)


def init_linear(module: nn.Module, gain: float) -> None:
    _init_linear_layers_shared(module, gain=float(gain))
