def register_all() -> None:
    from rl.algos import torchrl_ppo, torchrl_sac
    from rl.algos.registry import available_algos

    if "ppo" not in available_algos():
        torchrl_ppo.register()
    if "sac" not in available_algos():
        torchrl_sac.register()
