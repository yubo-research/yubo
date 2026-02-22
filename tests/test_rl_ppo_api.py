def test_ppo_register_delegates_to_registry(monkeypatch):
    """BO-style: import inside test to defer heavy deps (matches test_turbo_ackley)."""
    from rl.algos.backends.torchrl.ppo import api as ppo

    calls = []

    def fake_register_algo(name, config_cls, train_fn):
        calls.append((name, config_cls, train_fn))

    monkeypatch.setattr("rl.algos.registry.register_algo", fake_register_algo)
    ppo.register()

    assert len(calls) == 1
    name, config_cls, train_fn = calls[0]
    assert name == "ppo"
    assert config_cls is ppo.PPOConfig
    assert train_fn is ppo.train_ppo


def test_tanhnormal_support_property():
    """BO-style: import inside test to defer heavy deps."""
    from rl.algos.backends.torchrl.ppo import core as ppo

    dist = ppo._TanhNormal(
        loc=ppo.torch.zeros(1),
        scale=ppo.torch.ones(1),
    )
    assert dist.support is ppo.torch.distributions.constraints.real
