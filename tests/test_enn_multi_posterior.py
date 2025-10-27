import numpy as np


def test_multi_posterior_shapes_and_monotonicity():
    from model.enn import ENNNormal, EpistemicNearestNeighbors

    x = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=np.float32)
    y = np.array([[0.0], [0.0625], [0.25], [0.5625], [1.0]], dtype=np.float32)
    q = np.array([[0.1]], dtype=np.float32)

    enn = EpistemicNearestNeighbors(k=4, small_world_M=None)
    enn.add(x, y)

    ks = [1, 2, 3, 4]
    mp = enn.multi_posterior(q, ks=ks)

    assert isinstance(mp, ENNNormal)
    assert isinstance(mp.mu, np.ndarray)
    assert isinstance(mp.se, np.ndarray)
    assert mp.mu.shape == (1, len(ks), 1)
    assert mp.se.shape == (1, len(ks), 1)

    se = mp.se[0, :, 0]
    assert np.all(np.isfinite(se))


def test_multi_posterior_at_scale():
    from model.enn import ENNNormal, EpistemicNearestNeighbors

    num_obs = 100
    num_dim = 4
    num_metrics = 5
    ks = [3, 10]
    num_query = 6

    x = np.random.uniform(size=(num_obs, num_dim)).astype(np.float32)
    y = np.random.uniform(size=(num_obs, num_metrics)).astype(np.float32)
    q = np.random.uniform(size=(num_query, num_dim)).astype(np.float32)

    enn = EpistemicNearestNeighbors(k=max(ks), small_world_M=None)
    enn.add(x, y)

    mp = enn.multi_posterior(q, ks=ks)

    assert isinstance(mp, ENNNormal)
    assert isinstance(mp.mu, np.ndarray)
    assert isinstance(mp.se, np.ndarray)
    assert mp.mu.shape == (num_query, len(ks), num_metrics)
    assert mp.se.shape == (num_query, len(ks), num_metrics)


def test_multi_posterior_cumulative_slices():
    from model.enn import ENNNormal, EpistemicNearestNeighbors

    num_obs = 100
    num_dim = 4
    num_metrics = 5
    ks = [3, 10, 30]
    num_query = 6

    np.random.seed(42)
    x = np.random.uniform(size=(num_obs, num_dim)).astype(np.float32)
    y = np.random.uniform(size=(num_obs, num_metrics)).astype(np.float32)
    q = np.random.uniform(size=(num_query, num_dim)).astype(np.float32)

    enn_max = EpistemicNearestNeighbors(k=max(ks), small_world_M=None)
    enn_max.add(x, y)

    mp = enn_max.multi_posterior(q, ks=ks)

    assert isinstance(mp, ENNNormal)
    assert isinstance(mp.mu, np.ndarray)
    assert isinstance(mp.se, np.ndarray)
    assert mp.mu.shape == (num_query, len(ks), num_metrics)
    assert mp.se.shape == (num_query, len(ks), num_metrics)

    idx_all, dist_all = enn_max.about_neighbors(q, k=max(ks))

    y_slice0 = y[idx_all[:, 0:3]]
    y_slice1 = y[idx_all[:, 3:10]]
    y_slice2 = y[idx_all[:, 10:30]]

    dist_slice0 = dist_all[:, 0:3]
    dist_slice1 = dist_all[:, 3:10]
    dist_slice2 = dist_all[:, 10:30]

    mu_expected_0 = enn_max._calc_enn_normal(dist_slice0, y_slice0).mu
    mu_expected_1 = enn_max._calc_enn_normal(dist_slice1, y_slice1).mu
    mu_expected_2 = enn_max._calc_enn_normal(dist_slice2, y_slice2).mu

    assert np.allclose(mp.mu[:, 0, :], mu_expected_0)
    assert np.allclose(mp.mu[:, 1, :], mu_expected_1)
    assert np.allclose(mp.mu[:, 2, :], mu_expected_2)
