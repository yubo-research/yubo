import numpy as np
from enn.turbo.config.optimizer_config import OptimizerConfig

from optimizer.enn_turbo_optimizer import (
    TurboOptimizer,
    create_optimizer,
    predict_mu_sigma,
    scalarize,
)


def test_create_optimizer_and_helpers_smoke():
    cfg = OptimizerConfig()
    bounds = np.array([[-1.0, 1.0], [-2.0, 2.0]], dtype=float)
    opt = create_optimizer(bounds=bounds, config=cfg, rng=np.random.default_rng(0))
    assert isinstance(opt, TurboOptimizer)

    assert opt.tr_obs_count == 0
    assert opt.tr_length > 0.0
    _ = opt.init_progress

    mu_sigma = predict_mu_sigma(opt, np.zeros((1, 2), dtype=float))
    if mu_sigma is not None:
        mu, sigma = mu_sigma
        assert mu.shape == sigma.shape
        assert mu.shape[0] == 1

    scalar = scalarize(opt, np.zeros((1, 1), dtype=float))
    if scalar is not None:
        scalar = np.asarray(scalar, dtype=float).reshape(-1)
        assert scalar.shape == (1,)
