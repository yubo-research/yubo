def test_ppo_register_delegates_to_registry():
    from rl import builtins
    from rl.ppo.config import PPOConfig
    from rl.ppo.engine import train_ppo_puffer
    from rl.registry import get_algo

    builtins.register_all()
    spec = get_algo("ppo")
    assert spec.config_cls is PPOConfig
    assert spec.train_fn is train_ppo_puffer
