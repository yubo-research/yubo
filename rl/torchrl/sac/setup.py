from __future__ import annotations

import types

from rl.core import runtime as _torchrl_common
from rl.core.env_setup import build_continuous_gym_env_setup
from rl.core.sac_update import sac_update_step

from . import sac_setup_build as _sac_setup_build
from .sac_setup_action import _scale_action_to_env, _unscale_action_from_env
from .sac_setup_build import (
    build_modules,
    build_training,
    sac_update_shared,
)
from .sac_setup_collect_env import _make_collect_env_sac
from .sac_setup_models import _ActorNet, _QNet, _QNetPixel, _ScaleActionToEnv
from .sac_setup_types import _EnvSetup, _Modules, _TrainingSetup, _TrainState


sac_deps = types.SimpleNamespace(torchrl_common=_torchrl_common)
_DEFAULT_BUILD_CONTINUOUS = build_continuous_gym_env_setup


def build_env_setup(config):
    old_build_continuous = _sac_setup_build.build_continuous_gym_env_setup
    old_torchrl_common = _sac_setup_build.torchrl_common
    if build_continuous_gym_env_setup is not _DEFAULT_BUILD_CONTINUOUS:
        _sac_setup_build.build_continuous_gym_env_setup = build_continuous_gym_env_setup
    _sac_setup_build.torchrl_common = sac_deps.torchrl_common
    try:
        return _sac_setup_build.build_env_setup(config)
    finally:
        _sac_setup_build.build_continuous_gym_env_setup = old_build_continuous
        _sac_setup_build.torchrl_common = old_torchrl_common


__all__ = [
    "_ActorNet",
    "_EnvSetup",
    "_Modules",
    "_QNet",
    "_QNetPixel",
    "_ScaleActionToEnv",
    "_TrainingSetup",
    "_TrainState",
    "_make_collect_env_sac",
    "_scale_action_to_env",
    "_unscale_action_from_env",
    "build_continuous_gym_env_setup",
    "build_env_setup",
    "build_modules",
    "build_training",
    "sac_deps",
    "sac_update_shared",
    "sac_update_step",
]
