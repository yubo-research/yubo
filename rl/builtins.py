import rl.pufferlib as _pufferlib  # noqa: F401 - keeps pufferlib package in dependency graph
import rl.torchrl as _torchrl  # noqa: F401 - keeps torchrl package in dependency graph


def register_all() -> None:
    import problems.env_conf_atari_dm  # noqa: F401 - register atari/dm handlers for env_tag
    from rl.registry import (
        available_algos,
        register_algo_backend,
        register_algo_lazy,
    )

    if "ppo" not in available_algos():
        register_algo_lazy("ppo", "rl.torchrl.ppo.api")
    if "sac" not in available_algos():
        register_algo_lazy("sac", "rl.torchrl.sac.api")
    if "ppo_puffer" not in available_algos():
        register_algo_lazy("ppo_puffer", "rl.pufferlib.ppo.api")
    if "tdmpc2" not in available_algos():
        register_algo_lazy("tdmpc2", "rl.torchrl.tdmpc2.api")
    if "r2d2" not in available_algos():
        register_algo_lazy("r2d2", "rl.pufferlib.r2d2.api")

    register_algo_backend("ppo", "torchrl", "ppo")
    register_algo_backend("ppo", "pufferlib", "ppo_puffer")
    register_algo_backend("sac", "torchrl", "sac")
    register_algo_backend("tdmpc2", "torchrl", "tdmpc2")
    register_algo_backend("r2d2", "pufferlib", "r2d2")
