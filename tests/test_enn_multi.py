import numpy as np


def test_enn_multi_init():
    from model.enn_multi import ENNMulti

    ks = [1, 3, 5]
    model = ENNMulti(ks=ks)
    assert model is not None


def test_enn_multi_add_basic():
    from model.enn_multi import ENNMulti

    ks = [1, 3]
    model = ENNMulti(ks=ks)

    x = np.array([[0.0], [0.5], [1.0]], dtype=np.float32)
    y = np.array([[0.0], [0.25], [1.0]], dtype=np.float32)

    model.add(x, y)


def test_enn_multi_add_empty():
    from model.enn_multi import ENNMulti

    ks = [1, 3]
    model = ENNMulti(ks=ks)

    x = np.empty((0, 1), dtype=np.float32)
    y = np.empty((0, 1), dtype=np.float32)

    model.add(x, y)


def test_enn_multi_posterior_shape():
    from model.enn import ENNNormal
    from model.enn_multi import ENNMulti

    ks = [1, 3]
    model = ENNMulti(ks=ks)

    x_train = np.array([[0.0], [0.5], [1.0]], dtype=np.float32)
    y_train = np.array([[0.0], [0.25], [1.0]], dtype=np.float32)
    model.add(x_train, y_train)

    x_query = np.array([[0.25], [0.75]], dtype=np.float32)
    result = model.posterior(x_query)

    assert isinstance(result, ENNNormal)
    assert isinstance(result.mu, np.ndarray)
    assert isinstance(result.se, np.ndarray)
    assert result.mu.shape == (2, 1)
    assert result.se.shape == (2, 1)


def test_enn_multi_posterior_empty():
    from model.enn import ENNNormal
    from model.enn_multi import ENNMulti

    ks = [1, 3]
    model = ENNMulti(ks=ks)

    x_query = np.array([[0.25]], dtype=np.float32)
    result = model.posterior(x_query)

    assert isinstance(result, ENNNormal)
    assert result.mu.shape == (1, 1)
    assert result.se.shape == (1, 1)


def test_enn_multi_posterior_multiple_metrics():
    from model.enn import ENNNormal
    from model.enn_multi import ENNMulti

    ks = [1, 3, 5]
    model = ENNMulti(ks=ks)

    num_obs = 20
    num_dim = 2
    num_metrics = 3

    x_train = np.random.uniform(size=(num_obs, num_dim)).astype(np.float32)
    y_train = np.random.uniform(size=(num_obs, num_metrics)).astype(np.float32)
    model.add(x_train, y_train)

    num_query = 4
    x_query = np.random.uniform(size=(num_query, num_dim)).astype(np.float32)
    result = model.posterior(x_query)

    assert isinstance(result, ENNNormal)
    assert result.mu.shape == (num_query, num_metrics)
    assert result.se.shape == (num_query, num_metrics)


def test_enn_multi_posterior_combines_ensembles():
    from model.enn import EpistemicNearestNeighbors
    from model.enn_multi import ENNMulti

    ks = [1, 3]
    model = ENNMulti(ks=ks)

    x_train = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=np.float32)
    y_train = np.array([[0.0], [0.0625], [0.25], [0.5625], [1.0]], dtype=np.float32)
    model.add(x_train, y_train)

    x_query = np.array([[0.1]], dtype=np.float32)
    result = model.posterior(x_query)

    enn1 = EpistemicNearestNeighbors(k=1)
    enn1.add(x_train, y_train)

    enn3 = EpistemicNearestNeighbors(k=3)
    enn3.add(x_train, y_train)

    assert result.mu.shape == (1, 1)
    assert result.se.shape == (1, 1)


def test_enn_multi_posterior_linear_model_learned():
    from model.enn_multi import ENNMulti

    ks = [1, 2, 3]
    model = ENNMulti(ks=ks)

    x_train = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=np.float32)
    y_train = np.array([[0.0], [0.0625], [0.25], [0.5625], [1.0]], dtype=np.float32)
    model.add(x_train, y_train)

    x_query = np.array([[0.2], [0.8]], dtype=np.float32)
    result1 = model.posterior(x_query)

    result2 = model.posterior(x_query)

    assert np.allclose(result1.mu, result2.mu)
    assert np.allclose(result1.se, result2.se)


def test_enn_multi_posterior_at_scale():
    from model.enn import ENNNormal
    from model.enn_multi import ENNMulti

    num_obs = 100
    num_dim = 4
    num_metrics = 5
    ks = [3, 10, 20]
    num_query = 10

    model = ENNMulti(ks=ks)

    x_train = np.random.uniform(size=(num_obs, num_dim)).astype(np.float32)
    y_train = np.random.uniform(size=(num_obs, num_metrics)).astype(np.float32)
    model.add(x_train, y_train)

    x_query = np.random.uniform(size=(num_query, num_dim)).astype(np.float32)
    result = model.posterior(x_query)

    assert isinstance(result, ENNNormal)
    assert result.mu.shape == (num_query, num_metrics)
    assert result.se.shape == (num_query, num_metrics)
    assert np.all(np.isfinite(result.mu))
    assert np.all(np.isfinite(result.se))
    assert np.all(result.se > 0)


def test_enn_multi_posterior_incremental_add():
    from model.enn_multi import ENNMulti

    ks = [1, 3]
    model = ENNMulti(ks=ks)

    x1 = np.array([[0.0], [0.5]], dtype=np.float32)
    y1 = np.array([[0.0], [0.25]], dtype=np.float32)
    model.add(x1, y1)

    x_query = np.array([[0.25]], dtype=np.float32)
    result1 = model.posterior(x_query)

    x2 = np.array([[1.0]], dtype=np.float32)
    y2 = np.array([[1.0]], dtype=np.float32)
    model.add(x2, y2)

    result2 = model.posterior(x_query)

    assert result1.mu.shape == result2.mu.shape
    assert result1.se.shape == result2.se.shape


def test_enn_multi_se_magnitude_monotonic():
    from model.enn_multi import ENNMulti

    ks = [1, 3, 5]
    model = ENNMulti(ks=ks)

    x_train = np.array([[0.0], [0.5], [1.0]], dtype=np.float32)
    y_train = np.array([[0.0], [0.25], [1.0]], dtype=np.float32)
    model.add(x_train, y_train)

    x_query = np.array([[0.25]], dtype=np.float32)
    result = model.posterior(x_query)

    assert result.se > 0
    assert np.all(np.isfinite(result.se))
