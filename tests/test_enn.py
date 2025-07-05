import pytest


def set_up_enn(num_metrics):
    import numpy as np

    from model.enn import EpistemicNearestNeighbors

    num_dim = 5
    n = 10
    train_x = np.random.uniform(size=(n, num_dim))
    train_y = np.random.normal(size=(n, num_metrics))

    train_x[1] = train_x[2]
    train_y[1] = train_y[2]

    k = 3

    enn = EpistemicNearestNeighbors(train_x, train_y, k=k)
    return num_dim, n, train_x, train_y, k, enn


@pytest.mark.parametrize("num_metrics", [1, 2])
def test_enn_posterior(num_metrics):
    import numpy as np

    from model.enn import ENNNormal

    num_dim, n, train_x, train_y, k, enn = set_up_enn(num_metrics)

    num_dim = 3
    ennn = ENNNormal(mu=np.random.normal(size=(1, num_dim)), se=np.random.uniform(size=(1, num_dim)))
    assert ennn.sample(num_samples=1).shape == (1, num_dim, 1)
    assert ennn.sample(num_samples=2).shape == (1, num_dim, 2)
    assert ennn.sample(num_samples=100).shape == (1, num_dim, 100)


@pytest.mark.parametrize("num_metrics", [1, 2])
def test_add(num_metrics):
    import numpy as np

    num_dim, n, train_x, train_y, k, enn = set_up_enn(num_metrics)

    x = np.random.uniform(size=(1, num_dim))
    x_eps = x + 1e-6
    n = len(enn)
    idx_0, _ = enn.about_neighbors(x_eps)
    enn.add(x, 3 * np.ones(num_metrics))
    assert len(enn) == n + 1
    idx, _ = enn.about_neighbors(x_eps)
    assert np.all(idx_0 != idx)
    assert idx[0][0] == n


@pytest.mark.parametrize("num_metrics", [1, 2])
def test_enn(num_metrics):
    import numpy as np

    from model.enn import EpistemicNearestNeighbors

    num_dim, n, train_x, train_y, k, enn = set_up_enn(num_metrics)

    x = np.random.uniform(size=(1, num_dim))

    # If duplicates, you just get the first one
    #  so that knn_tools.farthest_neighbor()
    #  functions properly.
    assert np.all(enn.idx_x_slow(train_x[[1]]) == [1])
    assert enn.idx_x_slow(train_x[[3]]) == 3
    assert len(enn.idx_x_slow(x)) == 1
    assert np.all(enn.idx_x_slow(x) == [None])

    assert len(enn.about_neighbors(x)[0].flatten()) == k
    assert enn.neighbors(x).shape == (1, k, num_dim)
    assert len(enn.about_neighbors(x, k=1)[0].flatten()) == 1
    assert len(enn.about_neighbors(train_x[[0]], k=1)[0].flatten()) == 1
    assert len(enn.about_neighbors(train_x[[1]], k=1)[0].flatten()) == 1
    assert enn.neighbors(x, k=1).shape == (1, 1, num_dim)

    enn = EpistemicNearestNeighbors(np.empty((0, num_dim)), np.empty((0, num_metrics)), k=3)
    assert len(enn.about_neighbors(x)[0].flatten()) == 0
    assert enn.neighbors(x).shape == (0, num_dim)
    assert len(enn.about_neighbors(x, k=1)[0].flatten()) == 0
    assert enn.neighbors(x, k=1).shape == (0, num_dim)


@pytest.mark.parametrize("num_metrics", [1, 2])
def test_exclude_self(num_metrics):
    import numpy as np

    num_dim, n, train_x, train_y, k, enn = set_up_enn(num_metrics)

    x = train_x[[0, 5]]

    mvn_a = enn.posterior(x, exclude_nearest=False)
    mvn_b = enn.posterior(x, exclude_nearest=True)

    assert np.all(mvn_a.mu != mvn_b.mu)
    assert np.abs(mvn_a.mu - train_y[[0, 5]]).max() < 1e-6
    assert np.all(mvn_a.se < mvn_b.se / 1000)
