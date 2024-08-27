def test_normalizer():
    import numpy as np

    from problems.normalizer import Normalizer

    msf = Normalizer(shape=(3,))

    mu = np.array([0, 1.141, 3.14])
    sg = np.array([0.1, 0.2, 0.3])
    for _ in range(1000):
        x = mu + sg * np.random.normal(size=(3,))
        msf.update(x)

    m = msf.mean()
    s = msf.std()

    print(m)
    print(s)

    assert np.all(np.abs(m - mu) < 1e-1)
    assert np.all(np.abs(s - sg) < 1e-1)
