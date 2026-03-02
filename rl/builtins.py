import rl.pufferlib as _pufferlib  # noqa: F401 - keeps pufferlib package in dependency graph
import rl.torchrl as _torchrl  # noqa: F401 - keeps torchrl package in dependency graph


def _register_lazy_once(name: str, module_path: str, *, backend: str | None = None) -> None:
    from rl.registry import register_algo_lazy

    try:
        register_algo_lazy(name, module_path, backend=backend)
    except ValueError:
        # Idempotent re-registration during repeated CLI invocations.
        pass


def register_all() -> None:
    import problems.env_conf_atari_dm  # noqa: F401 - register atari/dm handlers for env_tag
    from rl.registry import (
        register_algo_backend,
    )

    _register_lazy_once("ppo", "rl.torchrl.ppo.core")
    _register_lazy_once("ppo", "rl.pufferlib.ppo.engine", backend="pufferlib")
    _register_lazy_once("sac", "rl.torchrl.sac.trainer")
    _register_lazy_once("sac", "rl.pufferlib.sac.engine", backend="pufferlib")

    register_algo_backend("ppo", "torchrl", "ppo")
    register_algo_backend("ppo", "pufferlib", "ppo")
    register_algo_backend("sac", "torchrl", "sac")
    register_algo_backend("sac", "pufferlib", "sac")
