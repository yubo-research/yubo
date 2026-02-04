def register_all() -> None:
    from rl.algos import ppo
    from rl.algos.registry import available_algos

    if "ppo" not in available_algos():
        ppo.register()
