def test_enn():
    import numpy as np

    from model.enn import EpsitemicNearestNeighbors

    num_dim = 5
    n = 10
    train_x = np.random.uniform(size=(n, num_dim))
    train_y = np.random.normal(size=(n, 1))

    train_x[1] = train_x[2]
    train_y[1] = train_y[2]

    x = np.random.uniform(size=(1, num_dim))
    k = 3

    enn = EpsitemicNearestNeighbors(train_x, train_y, k=k)

    assert np.all(enn.idx_x(train_x[1]) == [1, 2])
    assert enn.idx_x(train_x[3]) == 3
    assert len(enn.idx_x(x)) == 0

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
