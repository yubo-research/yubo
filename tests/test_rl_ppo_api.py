def test_ppo_register_delegates_to_registry(monkeypatch):
    from rl.torchrl.ppo import PPOConfig
    from rl.torchrl.ppo.core import register, train_ppo

    calls = []

    def fake_register_algo(name, config_cls, train_fn):
        calls.append((name, config_cls, train_fn))

    monkeypatch.setattr("rl.registry.register_algo", fake_register_algo)
    register()

    assert len(calls) == 1
    name, config_cls, train_fn = calls[0]
    assert name == "ppo"
    assert config_cls is PPOConfig
    assert train_fn is train_ppo


def test_tanhnormal_support_property():
    from rl.torchrl.ppo import core as ppo

    dist = ppo._TanhNormal(
        loc=ppo.torch.zeros(1),
        scale=ppo.torch.ones(1),
    )
    assert dist.support is ppo.torch.distributions.constraints.real
