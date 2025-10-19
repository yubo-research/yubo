def test_turbo_yubo_config_model_factory_contract():
    import torch

    from acq.turbo_yubo_config import TurboYUBOConfig

    cfg = TurboYUBOConfig()
    assert hasattr(cfg, "model_factory")
    factory = cfg.model_factory
    assert callable(factory)

    train_x = torch.rand(3, 2, dtype=torch.double)
    train_y = torch.rand(3, dtype=torch.double)
    model = factory(train_x=train_x, train_y=train_y)

    assert hasattr(model, "train_inputs")
    assert hasattr(model, "train_targets")
    assert hasattr(model, "covar_module")

    test_x = torch.rand(2, 2, dtype=torch.double)
    posterior = model.posterior(test_x)
    assert hasattr(posterior, "sample")
