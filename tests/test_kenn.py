def test_kenn_basic():
    import numpy as np

    from model.kenn import KernlizedEpistemicNearestNeighbors

    num_dim = 5
    n = 10
    train_x = np.random.uniform(size=(n, num_dim))
    train_y = np.random.normal(size=(n, 1))

    lengthscales = np.ones(num_dim)
    k = 3

    kenn = KernlizedEpistemicNearestNeighbors(lengthscales=lengthscales, k=k)
    kenn.add(train_x, train_y)

    assert len(kenn) == n

    x = np.random.uniform(size=(2, num_dim))
    mvn = kenn.posterior(x)

    assert mvn.mu.shape == (2, 1)
    assert mvn.se.shape == (2, 1)
    assert np.all(np.isfinite(mvn.mu))
    assert np.all(np.isfinite(mvn.se))
    assert np.all(mvn.se > 0)


def test_kenn_empty():
    import numpy as np

    from model.kenn import KernlizedEpistemicNearestNeighbors

    num_dim = 3
    lengthscales = np.ones(num_dim)
    kenn = KernlizedEpistemicNearestNeighbors(lengthscales=lengthscales, k=3)

    x = np.random.uniform(size=(2, num_dim))
    mvn = kenn.posterior(x)

    assert mvn.mu.shape == (2, 1)
    assert mvn.se.shape == (2, 1)
    assert np.all(mvn.mu == 0)
    assert np.all(mvn.se == 1)


def test_kenn_add():
    import numpy as np

    from model.kenn import KernlizedEpistemicNearestNeighbors

    num_dim = 4
    lengthscales = np.ones(num_dim)
    kenn = KernlizedEpistemicNearestNeighbors(lengthscales=lengthscales, k=3)

    assert len(kenn) == 0

    train_x = np.random.uniform(size=(5, num_dim))
    train_y = np.random.normal(size=(5, 1))
    kenn.add(train_x, train_y)

    assert len(kenn) == 5

    train_x2 = np.random.uniform(size=(3, num_dim))
    train_y2 = np.random.normal(size=(3, 1))
    kenn.add(train_x2, train_y2)

    assert len(kenn) == 8


def test_kenn_sample():
    import numpy as np

    from model.kenn import KernlizedEpistemicNearestNeighbors

    num_dim = 3
    n = 10
    train_x = np.random.uniform(size=(n, num_dim))
    train_y = np.random.normal(size=(n, 1))

    lengthscales = np.ones(num_dim)
    kenn = KernlizedEpistemicNearestNeighbors(lengthscales=lengthscales, k=3)
    kenn.add(train_x, train_y)

    x = np.random.uniform(size=(2, num_dim))
    mvn = kenn.posterior(x)

    samples = mvn.sample(num_samples=5)
    assert samples.shape == (2, 1, 5)
    assert np.all(np.isfinite(samples))


def test_kenn_lengthscales():
    import numpy as np

    from model.kenn import KernlizedEpistemicNearestNeighbors

    num_dim = 2
    n = 10
    train_x = np.random.uniform(size=(n, num_dim))
    train_y = np.random.normal(size=(n, 1))

    lengthscales = np.array([0.1, 1.0])
    kenn = KernlizedEpistemicNearestNeighbors(lengthscales=lengthscales, k=3)
    kenn.add(train_x, train_y)

    x = np.random.uniform(size=(1, num_dim))
    mvn = kenn.posterior(x)

    assert mvn.mu.shape == (1, 1)
    assert mvn.se.shape == (1, 1)
