def test_calculate_sobol_indices_linear_additive():
    import numpy as np

    from sampling.sobol_indices import calculate_sobol_indices_np

    rng = np.random.default_rng(0)
    n = 6000
    d = 5
    X = rng.uniform(0.0, 1.0, size=(n, d))
    a1 = 2.0
    a2 = 1.0
    noise_sd = 0.05
    y = a1 * X[:, 0] + a2 * X[:, 1] + rng.normal(0.0, noise_sd, size=n)

    S = calculate_sobol_indices_np(X, y)

    vx = 1.0 / 12.0
    vy = a1 * a1 * vx + a2 * a2 * vx + noise_sd * noise_sd
    s1_true = a1 * a1 * vx / vy
    s2_true = a2 * a2 * vx / vy

    assert S.shape == (d,)
    assert np.isfinite(S).all()
    assert abs(S[0] - s1_true) < 0.07
    assert abs(S[1] - s2_true) < 0.07
    assert (S[2:] < 0.1).all()
    assert S.min() >= 0.0
    assert S.max() <= 1.0 + 1e-6


def test_calculate_sobol_indices_nonlinear_single_input():
    import numpy as np

    from sampling.sobol_indices import calculate_sobol_indices_np

    rng = np.random.default_rng(1)
    n = 5000
    d = 3
    X = rng.uniform(0.0, 1.0, size=(n, d))
    noise_sd = 0.1
    y = X[:, 0] ** 2 + rng.normal(0.0, noise_sd, size=n)

    S = calculate_sobol_indices_np(X, y)

    vx2 = 1.0 / 5.0 - (1.0 / 3.0) ** 2
    vy = vx2 + noise_sd * noise_sd
    s1_true = vx2 / vy

    assert S.shape == (d,)
    assert abs(S[0] - s1_true) < 0.1
    assert (S[1:] < 0.1).all()


def test_calculate_sobol_indices_small_n():
    import numpy as np

    from sampling.sobol_indices import calculate_sobol_indices_np

    rng = np.random.default_rng(7)
    d = 4

    X1 = rng.uniform(0.0, 1.0, size=(1, d))
    y1 = rng.normal(0.0, 1.0, size=1)

    S1 = calculate_sobol_indices_np(X1, y1)
    assert S1.shape == (d,)
    assert np.allclose(S1, np.ones(d) / d)

    for n in [2, 3, 10, 30]:
        X = rng.uniform(0.0, 1.0, size=(n, d))
        y = X[:, 0] + 0.1 * rng.normal(0.0, 1.0, size=n)
        S = calculate_sobol_indices_np(X, y)
        assert S.shape == (d,)
        assert np.isfinite(S).all()
        assert (S >= 0.0).all()
        assert (S <= 1.0 + 1e-12).all()
