import numpy as np


def test_cov_diag_multiple_points_mean_squared_deviation():
    from sampling.x_cov import cov_diag

    x0 = np.array([1.0, -1.0])
    X = np.array(
        [
            [2.0, -2.0],
            [0.0, -1.0],
            [1.0, 3.0],
        ]
    )
    C = cov_diag(x0, X)
    diff2 = (X - x0) ** 2
    v = diff2.mean(axis=0)
    expected = v
    assert C.shape == (2,)
    assert np.allclose(C, expected)


def test_cov_diag_shape_mismatch_raises():
    from sampling.x_cov import cov_diag

    x0 = np.zeros(3)
    X = np.ones((5, 2))
    try:
        _ = cov_diag(x0, X)
        raised = False
    except AssertionError:
        raised = True
    assert raised


def test_evec_1_matches_full_covariance():
    from sampling.x_cov import evec_1

    rng = np.random.default_rng(0)
    x0 = np.array([0.5, -0.25])
    X = rng.normal(size=(50, 2))
    Xc = X - x0
    C = (Xc.T @ Xc) / float(X.shape[0])
    w, V = np.linalg.eigh(C)
    v_ref = V[:, -1]
    v = evec_1(x0, X)
    if np.dot(v, v_ref) < 0:
        v = -v
    assert np.allclose(v, v_ref, atol=1e-6)


def test_evec_1_zero_variance_returns_zero_vector():
    from sampling.x_cov import evec_1

    x0 = np.array([1.0, 2.0, 3.0])
    X = np.tile(x0, (5, 1))
    v = evec_1(x0, X)
    assert np.allclose(v, np.zeros_like(x0))


def test_evec_1_is_much_faster_than_eigh_full_covariance():
    import time

    from sampling.x_cov import evec_1

    rng = np.random.default_rng(1)
    n = 1024
    d = 512
    X = rng.normal(size=(n, d))
    x0 = rng.normal(size=(d,))

    reps = 3
    t_evec = []
    t_eigh = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _ = evec_1(x0, X)
        t1 = time.perf_counter()
        centered = X - x0
        t2 = time.perf_counter()
        C = (centered.T @ centered) / float(n)
        _w, _V = np.linalg.eigh(C)
        t3 = time.perf_counter()
        t_evec.append(t1 - t0)
        t_eigh.append(t3 - t2)

    m_evec = float(np.median(t_evec))
    m_eigh = float(np.median(t_eigh))
    assert m_evec < (m_eigh / 5.0)
