def test_ms_filter():
    import numpy as np

    from problems.ms_filter import MeanStdFilter

    msf = MeanStdFilter(shape=(3,))

    mu = np.array([0, 1.141, 3.14])
    sg = np.array([0.1, 0.2, 0.3])
    for _ in range(1000):
        x = mu + sg * np.random.normal(size=(3,))
        msf(x)
    m, s = msf.get_stats()

    assert np.all(np.abs(m - mu) < 1e-1)
    assert np.all(np.abs(s - sg) < 1e-1)
