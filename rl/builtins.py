def _register_lazy_once(name: str, module_path: str) -> None:
    from rl.registry import register_algo_lazy

    try:
        register_algo_lazy(name, module_path)
    except ValueError:
        pass


def register_all() -> None:
    _register_lazy_once("ppo", "rl.ppo.engine")
    _register_lazy_once("sac", "rl.sac.engine")
