from __future__ import annotations

from rl.torchrl.collect_utils import make_collect_env

from .sac_setup_types import _EnvSetup


def _make_collect_env_sac(
    env_conf,
    env_setup: _EnvSetup,
    env_index: int = 0,
    *,
    num_envs: int = 1,
    device=None,
):
    # env_setup is preserved for interface compatibility but seeds are handled via env_conf
    return make_collect_env(env_conf, env_index=env_index, num_envs=int(num_envs), device=device)
