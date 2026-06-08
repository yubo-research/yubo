from __future__ import annotations

import types

from rl.core import runtime
from rl.core.sac_update import sac_update_step

from . import sac_setup_build
from .sac_setup_action import _scale_action_to_env, _unscale_action_from_env
from .sac_setup_build import (
    build_modules,
    build_training,
    sac_update_shared,
)
from .sac_setup_collect_env import _make_collect_env_sac
from .sac_setup_models import _ActorNet, _QNet, _QNetPixel, _ScaleActionToEnv
from .sac_setup_types import _EnvSetup, _Modules, _TrainingSetup, _TrainState

sac_deps = types.SimpleNamespace(torchrl_common=runtime)


def build_env_setup(config):
    old_runtime = sac_setup_build.runtime
    sac_setup_build.runtime = sac_deps.torchrl_common
    try:
        return sac_setup_build.build_env_setup(config)
    finally:
        sac_setup_build.runtime = old_runtime


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
    "build_env_setup",
    "build_modules",
    "build_training",
    "sac_deps",
    "sac_update_shared",
    "sac_update_step",
]
