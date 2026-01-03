import numpy as np


def test_sigma_x_single_point_causes_invalid_indices():
    from model.enn_weighter import ENNWeighter

    x = np.array([[1.0, 2.0]], dtype=np.float32)
    y = np.array([[0.5]], dtype=np.float32)
    q = np.array([[1.5, 2.5]], dtype=np.float32)

    enn = ENNWeighter(k=1, small_world_M=None, weighting="sigma_x")
    enn.add(x, y)

    posterior = enn.posterior(q)
    assert np.isfinite(posterior.mu).all()
    assert np.isfinite(posterior.se).all()
