def _register_lazy_once(name: str, module_path: str, *, backend: str | None = None) -> None:
    from rl.registry import register_algo_lazy

    try:
        register_algo_lazy(name, module_path, backend=backend)
    except ValueError:
        pass


def register_all() -> None:
    import importlib

    from rl.registry import register_algo_backend

    importlib.import_module("problems.env_conf_atari_dm")
    _register_lazy_once("ppo", "rl.torchrl.ppo.core")
    _register_lazy_once("ppo", "rl.pufferlib.ppo.engine", backend="pufferlib")
    _register_lazy_once("sac", "rl.torchrl.sac.trainer")
    _register_lazy_once("sac", "rl.pufferlib.sac.engine", backend="pufferlib")
    register_algo_backend("ppo", "torchrl", "ppo")
    register_algo_backend("ppo", "pufferlib", "ppo")
    register_algo_backend("sac", "torchrl", "sac")
    register_algo_backend("sac", "pufferlib", "sac")
