from __future__ import annotations

from rl.torchrl.collect_utils import make_collect_env


def _make_collect_env(env_conf, env_index: int = 0, *, num_envs: int = 1, device=None):
    return make_collect_env(env_conf, env_index=int(env_index), num_envs=int(num_envs), device=device)


def _make_collect_env_factory(env_conf, num_envs: int):
    env_index = [0]

    def fn():
        idx = env_index[0]
        env_index[0] = (idx + 1) % num_envs
        return make_collect_env(env_conf, env_index=idx)

    return fn
