def test_aenn_single_observation_exact_mean():
    import numpy as np

    from model.aenn import AdditiveEpistemicNearestNeighbors

    np.random.seed(0)
    d = 3
    aenn = AdditiveEpistemicNearestNeighbors(k=3)
    x0 = np.random.uniform(size=(1, d))
    y0 = np.array([[2.5]])
    aenn.add(x0, y0)

    xq = np.random.uniform(size=(5, d))
    mvn = aenn.posterior(xq)

    assert mvn.mu.shape == (5, 1)
    assert np.allclose(mvn.mu, y0[None, :], atol=1e-12)


def test_aenn_basic():
    import numpy as np

    from model.aenn import AdditiveEpistemicNearestNeighbors

    num_dim = 5
    n = 20
    train_x = np.random.uniform(size=(n, num_dim))
    train_y = np.random.normal(size=(n, 1))

    k = 3
    aenn = AdditiveEpistemicNearestNeighbors(k=k)
    aenn.add(train_x, train_y)

    assert len(aenn) == n

    x = np.random.uniform(size=(2, num_dim))
    mvn = aenn.posterior(x)

    assert mvn.mu.shape == (2, 1)
    assert mvn.se.shape == (2, 1)
    assert np.all(np.isfinite(mvn.mu))
    assert np.all(np.isfinite(mvn.se))
    assert np.all(mvn.se > 0)


def test_aenn_empty():
    import numpy as np

    from model.aenn import AdditiveEpistemicNearestNeighbors

    num_dim = 3
    aenn = AdditiveEpistemicNearestNeighbors(k=3)

    x = np.random.uniform(size=(2, num_dim))
    mvn = aenn.posterior(x)

    assert mvn.mu.shape == (2, 1)
    assert mvn.se.shape == (2, 1)
    assert np.all(mvn.mu == 0)
    assert np.all(mvn.se == 1)


def test_aenn_add():
    import numpy as np

    from model.aenn import AdditiveEpistemicNearestNeighbors

    num_dim = 4
    aenn = AdditiveEpistemicNearestNeighbors(k=3)

    assert len(aenn) == 0

    train_x = np.random.uniform(size=(10, num_dim))
    train_y = np.random.normal(size=(10, 1))
    aenn.add(train_x, train_y)

    assert len(aenn) == 10

    train_x2 = np.random.uniform(size=(5, num_dim))
    train_y2 = np.random.normal(size=(5, 1))
    aenn.add(train_x2, train_y2)

    assert len(aenn) == 15


def test_aenn_sample():
    import numpy as np

    from model.aenn import AdditiveEpistemicNearestNeighbors

    num_dim = 3
    n = 20
    train_x = np.random.uniform(size=(n, num_dim))
    train_y = np.random.normal(size=(n, 1))

    aenn = AdditiveEpistemicNearestNeighbors(k=3)
    aenn.add(train_x, train_y)

    x = np.random.uniform(size=(2, num_dim))
    mvn = aenn.posterior(x)

    samples = mvn.sample(num_samples=5)
    assert samples.shape == (2, 1, 5)
    assert np.all(np.isfinite(samples))


def test_aenn_minimal_observations():
    import numpy as np

    from model.aenn import AdditiveEpistemicNearestNeighbors

    num_dim = 3
    aenn = AdditiveEpistemicNearestNeighbors(k=3)

    train_x = np.random.uniform(size=(3, num_dim))
    train_y = np.random.normal(size=(3, 1))
    aenn.add(train_x, train_y)

    x = np.random.uniform(size=(1, num_dim))
    mvn = aenn.posterior(x)

    assert mvn.mu.shape == (1, 1)
    assert mvn.se.shape == (1, 1)
    assert np.all(np.isfinite(mvn.mu))
    assert np.all(np.isfinite(mvn.se))


def test_aenn_fewer_than_k_observations():
    import numpy as np

    from model.aenn import AdditiveEpistemicNearestNeighbors

    num_dim = 3
    aenn = AdditiveEpistemicNearestNeighbors(k=5)

    train_x = np.random.uniform(size=(3, num_dim))
    train_y = np.random.normal(size=(3, 1))
    aenn.add(train_x, train_y)

    assert aenn._weights is not None
    assert aenn._weights.shape == (num_dim, 1)

    x = np.random.uniform(size=(2, num_dim))
    mvn = aenn.posterior(x)

    assert mvn.mu.shape == (2, 1)
    assert mvn.se.shape == (2, 1)
    assert np.all(np.isfinite(mvn.mu))
    assert np.all(np.isfinite(mvn.se))


def test_aenn_beta_learning():
    import numpy as np

    from model.aenn import AdditiveEpistemicNearestNeighbors

    num_dim = 2
    n = 15
    train_x = np.random.uniform(size=(n, num_dim))
    train_y = np.random.normal(size=(n, 1))

    aenn = AdditiveEpistemicNearestNeighbors(k=3)
    aenn.add(train_x, train_y)

    assert aenn._weights is not None
    assert aenn._weights.shape == (num_dim, 1)
    assert np.all(np.isfinite(aenn._weights))


def test_aenn_multi_output():
    import numpy as np

    from model.aenn import AdditiveEpistemicNearestNeighbors

    num_dim = 4
    n = 20
    train_x = np.random.uniform(size=(n, num_dim))
    train_y = np.random.normal(size=(n, 2))

    aenn = AdditiveEpistemicNearestNeighbors(k=3)
    aenn.add(train_x, train_y)

    x = np.random.uniform(size=(3, num_dim))
    mvn = aenn.posterior(x)

    assert mvn.mu.shape == (3, 2)
    assert mvn.se.shape == (3, 2)
    assert np.all(np.isfinite(mvn.mu))
    assert np.all(np.isfinite(mvn.se))


def test_aenn_consistency():
    import numpy as np

    from model.aenn import AdditiveEpistemicNearestNeighbors

    num_dim = 3
    n = 20
    np.random.seed(42)
    train_x = np.random.uniform(size=(n, num_dim))
    train_y = np.random.normal(size=(n, 1))

    aenn = AdditiveEpistemicNearestNeighbors(k=3)
    aenn.add(train_x, train_y)

    assert aenn._weights is not None
    assert aenn._weights.shape == (num_dim, 1)

    x1 = np.random.uniform(size=(1, num_dim))
    x2 = np.random.uniform(size=(1, num_dim))

    mvn1 = aenn.posterior(x1)
    mvn2 = aenn.posterior(x2)

    assert mvn1.mu.shape == (1, 1)
    assert mvn1.se.shape == (1, 1)
    assert mvn2.mu.shape == (1, 1)
    assert mvn2.se.shape == (1, 1)

    x_batch = np.vstack([x1, x2])
    mvn_batch = aenn.posterior(x_batch)

    assert np.allclose(mvn_batch.mu[0:1], mvn1.mu)
    assert np.allclose(mvn_batch.mu[1:2], mvn2.mu)
    assert np.allclose(mvn_batch.se[0:1], mvn1.se)
    assert np.allclose(mvn_batch.se[1:2], mvn2.se)
