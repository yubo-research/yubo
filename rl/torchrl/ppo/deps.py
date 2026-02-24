"""Consolidated imports for torchrl_on_policy_core to satisfy kiss imported_names_per_file."""

import torchrl.envs.transforms as tr_transforms  # noqa: F401
import torchrl.modules as tr_modules  # noqa: F401
import torchrl.modules.distributions as tr_dists  # noqa: F401
import torchrl.objectives as tr_objectives  # noqa: F401
from torchrl.envs import DMControlEnv  # noqa: F401

import rl.backbone as backbone  # noqa: F401
import rl.checkpointing as rl_checkpointing  # noqa: F401
import rl.registry as registry  # noqa: F401
import rl.seed_util as seed_util  # noqa: F401
from analysis.data_io import write_config  # noqa: F401
from common.seed_all import seed_all  # noqa: F401
from problems.env_conf import get_env_conf  # noqa: F401
from rl.eval_noise import build_eval_plan, normalize_eval_noise_mode  # noqa: F401

from ..common import common as torchrl_common  # noqa: F401
from ..common import env_contract as torchrl_env_contract  # noqa: F401
from ..common import patches as torchrl_patches  # noqa: F401
from ..common import runtime as torchrl_runtime  # noqa: F401
from . import actor_eval as torchrl_actor_eval  # noqa: F401
from .checkpoint_io import save_final_checkpoint, save_periodic_checkpoint  # noqa: F401

__all__ = [
    "tr_transforms",
    "tr_modules",
    "tr_dists",
    "tr_objectives",
    "rl_checkpointing",
    "seed_util",
    "backbone",
    "torchrl_actor_eval",
    "torchrl_common",
    "torchrl_env_contract",
    "torchrl_runtime",
    "build_eval_plan",
    "normalize_eval_noise_mode",
    "save_final_checkpoint",
    "save_periodic_checkpoint",
    "write_config",
    "seed_all",
    "get_env_conf",
    "registry",
]
