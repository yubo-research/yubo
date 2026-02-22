from rl.algos import backends as _backends  # noqa: F401 - keeps backend package in dependency graph
from rl.algos.backends import torchrl as _torchrl  # noqa: F401 - keeps torchrl package in dependency graph


def register_all() -> None:
    import problems.env_conf_atari_dm  # noqa: F401 - register atari/dm handlers for env_tag
    from rl.algos.registry import (
        available_algos,
        register_algo_backend,
        register_algo_lazy,
    )

    if "ppo" not in available_algos():
        register_algo_lazy("ppo", "rl.algos.backends.torchrl.ppo.api")
    if "sac" not in available_algos():
        register_algo_lazy("sac", "rl.algos.backends.torchrl.sac.api")
    if "ppo_puffer" not in available_algos():
        register_algo_lazy("ppo_puffer", "rl.algos.backends.pufferlib.ppo.api")

    register_algo_backend("ppo", "torchrl", "ppo")
    register_algo_backend("ppo", "pufferlib", "ppo_puffer")
    register_algo_backend("sac", "torchrl", "sac")
