def test_turbo_yubo_kenn_model_factory_contract():
    import torch

    from acq.turbo_yubo.ty_kenn_model_factory import build_turbo_yubo_kenn_model

    train_x = torch.rand(5, 3, dtype=torch.double)
    train_y = torch.rand(5, dtype=torch.double)

    model = build_turbo_yubo_kenn_model(train_x=train_x, train_y=train_y, k=3)

    assert hasattr(model, "train_inputs")
    assert hasattr(model, "train_targets")
    assert hasattr(model, "covar_module")

    test_x = torch.rand(4, 3, dtype=torch.double)
    posterior = model.posterior(test_x)
    s = posterior.sample(sample_shape=torch.Size([2]))
    assert s.shape[-2] == test_x.shape[-2]


def test_turbo_yubo_kenn_with_lengthscales():
    import numpy as np
    import torch

    from acq.turbo_yubo.ty_kenn_model_factory import build_turbo_yubo_kenn_model

    train_x = torch.rand(5, 3, dtype=torch.double)
    train_y = torch.rand(5, dtype=torch.double)

    lengthscales = np.array([0.1, 0.5, 1.0])
    model = build_turbo_yubo_kenn_model(train_x=train_x, train_y=train_y, k=3, lengthscales=lengthscales)

    assert hasattr(model, "covar_module")
    ls = model.covar_module.lengthscale
    assert ls.shape == (1, 3)
    assert torch.allclose(ls[0], torch.tensor([0.1, 0.5, 1.0], dtype=torch.float32))


def test_turbo_yubo_kenn_default_lengthscales():
    import torch

    from acq.turbo_yubo.ty_kenn_model_factory import build_turbo_yubo_kenn_model

    train_x = torch.rand(5, 4, dtype=torch.double)
    train_y = torch.rand(5, dtype=torch.double)

    model = build_turbo_yubo_kenn_model(train_x=train_x, train_y=train_y, k=3)

    assert hasattr(model, "covar_module")
    ls = model.covar_module.lengthscale
    assert ls.shape == (1, 4)
    assert torch.allclose(ls[0], torch.ones(4, dtype=torch.float32))


def test_turbo_yubo_designer_with_kenn_factory():
    import torch

    from acq.turbo_yubo.turbo_yubo_config import TurboYUBOConfig
    from acq.turbo_yubo.ty_kenn_model_factory import build_turbo_yubo_kenn_model
    from optimizer.turbo_yubo_designer import TurboYUBODesigner
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=11, noise_seed_0=12)
    policy = default_policy(env_conf)

    class _Datum:
        def __init__(self, policy, r):
            self.policy = policy

            class _T:
                pass

            self.trajectory = _T()
            self.trajectory.rreturn = float(r)

    data = []
    for i in range(4):
        p = policy.clone()
        data.append(_Datum(p, float(torch.rand(()).item())))

    cfg = TurboYUBOConfig()
    cfg.model_factory = build_turbo_yubo_kenn_model

    designer = TurboYUBODesigner(policy, config=cfg)
    out = designer(data, num_arms=3)
    assert len(out) == 3
