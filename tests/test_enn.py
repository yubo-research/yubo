def set_up_enn():
    import numpy as np

    from model.enn import EpsitemicNearestNeighbors

    num_dim = 5
    n = 10
    train_x = np.random.uniform(size=(n, num_dim))
    train_y = np.random.normal(size=(n, 1))

    train_x[1] = train_x[2]
    train_y[1] = train_y[2]

    k = 3

    enn = EpsitemicNearestNeighbors(train_x, train_y, k=k)
    return num_dim, n, train_x, train_y, k, enn


def test_enn_posterior():
    import numpy as np

    from model.enn import ENNNormal

    num_dim, n, train_x, train_y, k, enn = set_up_enn()

    num_dim = 3
    ennn = ENNNormal(mu=np.random.normal(size=(1, num_dim)), se=np.random.uniform(size=(1, num_dim)))
    assert ennn.sample(num_samples=1).shape == (1, num_dim, 1)
    assert ennn.sample(num_samples=2).shape == (1, num_dim, 2)
    assert ennn.sample(num_samples=100).shape == (1, num_dim, 100)


def test_add():
    import numpy as np

    num_dim, n, train_x, train_y, k, enn = set_up_enn()

    x = np.random.uniform(size=(1, num_dim))
    x_eps = x + 1e-6
    n = len(enn)
    idx_0, _ = enn.about_neighbors(x_eps)
    enn.add(x, 3)
    assert len(enn) == n + 1
    idx, _ = enn.about_neighbors(x_eps)
    assert np.all(idx_0 != idx)
    assert idx[0] == n


def test_enn():
    import numpy as np

    from model.enn import EpsitemicNearestNeighbors

    num_dim, n, train_x, train_y, k, enn = set_up_enn()

    x = np.random.uniform(size=(1, num_dim))

    # If duplicates, you just get the first one
    #  so that knn_tools.farthest_neighbor()
    #  functions properly.
    assert np.all(enn.idx_x(train_x[[1]]) == [1])
    assert enn.idx_x(train_x[[3]]) == 3
    assert len(enn.idx_x(x)) == 1
    assert np.all(enn.idx_x(x) == [None])

    assert len(enn.about_neighbors(x)[0]) == k
    assert enn.neighbors(x).shape == (k, num_dim)
    assert len(enn.about_neighbors(x, k=1)[0]) == 1
    assert len(enn.about_neighbors(train_x[[0]], k=1)[0]) == 1
    assert len(enn.about_neighbors(train_x[[1]], k=1)[0]) == 1
    assert enn.neighbors(x, k=1).shape == (1, num_dim)

    enn = EpsitemicNearestNeighbors(np.empty((0, num_dim)), np.empty((0, 1)), k=3)
    assert len(enn.about_neighbors(x)[0]) == 0
    assert enn.neighbors(x).shape == (0, num_dim)
    assert len(enn.about_neighbors(x, k=1)[0]) == 0
    assert enn.neighbors(x, k=1).shape == (0, num_dim)
