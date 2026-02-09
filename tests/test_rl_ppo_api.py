from rl.algos import torchrl_ppo


def test_train_result_dataclass_fields():
    result = torchrl_ppo.TrainResult(
        best_return=1.5,
        last_eval_return=1.0,
        last_heldout_return=None,
        num_iterations=3,
    )
    assert result.best_return == 1.5
    assert result.last_eval_return == 1.0
    assert result.last_heldout_return is None
    assert result.num_iterations == 3


def test_ppo_register_delegates_to_registry(monkeypatch):
    calls = []

    def fake_register_algo(name, config_cls, train_fn):
        calls.append((name, config_cls, train_fn))

    monkeypatch.setattr("rl.algos.registry.register_algo", fake_register_algo)
    torchrl_ppo.register()

    assert len(calls) == 1
    name, config_cls, train_fn = calls[0]
    assert name == "ppo"
    assert config_cls is torchrl_ppo.PPOConfig
    assert train_fn is torchrl_ppo.train_ppo


def test_tanhnormal_support_property():
    dist = torchrl_ppo._TanhNormal(
        loc=torchrl_ppo.torch.zeros(1),
        scale=torchrl_ppo.torch.ones(1),
    )
    assert dist.support is torchrl_ppo.torch.distributions.constraints.real
