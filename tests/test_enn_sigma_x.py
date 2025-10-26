import numpy as np


def test_sigma_x_changes_neighbor_identity_and_scales_by_inverse_std():
    from model.enn import EpistemicNearestNeighbors
    from model.enn_weighter import ENNWeighter

    x = np.array(
        [
            [10.0, 0.001],
            [0.5, 0.01],
            [1000.0, 0.0],
            [-1000.0, 0.0],
        ],
        dtype=np.float32,
    )
    y = np.zeros((x.shape[0], 1), dtype=np.float32)
    q = np.array([[0.0, 0.0]], dtype=np.float32)

    enn_none = EpistemicNearestNeighbors(k=1, small_world_M=None)
    enn_none.add(x, y)
    idx_none, _ = enn_none.about_neighbors(q, k=1)
    assert idx_none.shape == (1, 1)
    assert idx_none[0, 0] == 1

    enn_sig = ENNWeighter(k=1, small_world_M=None, weighting="sigma_x")
    enn_sig.add(x, y)
    enn_sig.set_x_center(None)
    idx_sig, _ = enn_sig.about_neighbors(q, k=1)
    assert idx_sig.shape == (1, 1)
    assert idx_sig[0, 0] == 0

    s = np.std(x, axis=0)
    r_expected = s[1] / s[0]
    r_actual = enn_sig._weights[0] / enn_sig._weights[1]
    assert np.isfinite(r_actual)
    np.testing.assert_allclose(r_actual, r_expected, rtol=0.2)


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
