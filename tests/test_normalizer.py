def test_normalizer():
    import numpy as np

    from problems.normalizer import Normalizer

    np.random.seed(0)
    msf = Normalizer(shape=(3,))

    mu = np.array([0, 1.141, 3.14])
    sg = np.array([0.1, 0.2, 0.3])
    for _ in range(1000):
        x = mu + sg * np.random.normal(size=(3,))
        msf.update(x)

    m, s = msf.mean_and_std()

    print(m)
    print(s)

    assert np.all(np.abs(m - mu) < 1e-1)
    assert np.all(np.abs(s - sg) < 1e-1)


def test_normalizer_decay_adapts():
    import numpy as np

    from problems.normalizer import Normalizer

    n_static = Normalizer(shape=(1,))
    n_decay = Normalizer(shape=(1,), decay=0.5)

    for _ in range(100):
        x = np.array([0.0])
        n_static.update(x)
        n_decay.update(x)

    for _ in range(10):
        x = np.array([10.0])
        n_static.update(x)
        n_decay.update(x)

    m_static, _ = n_static.mean_and_std()
    m_decay, _ = n_decay.mean_and_std()
    assert float(m_static[0]) < 2.0
    assert float(m_decay[0]) > 9.0
