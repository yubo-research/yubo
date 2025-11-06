def test_gp_ensemble_posterior_interpolates_and_zero_variance_at_train_point():
    import numpy as np

    from model.gp_ensemble import GPEnsemble

    n = 30
    d = 3
    rng = np.random.default_rng(123)
    train_x = rng.uniform(size=(n, d))
    train_y = rng.normal(size=(n, 1))

    model = GPEnsemble(train_x, train_y)

    i = 7
    xq = train_x[i : i + 1]
    mvn = model.posterior(xq, k=5, exclude_nearest=False)

    assert mvn.mu.shape == (1, 1)
    assert mvn.se.shape == (1, 1)
    assert np.allclose(mvn.mu[0, 0], train_y[i, 0], atol=1e-8)
    assert mvn.se[0, 0] <= 1e-8


def test_gp_ensemble_posterior_exclude_nearest_increases_variance():
    import numpy as np

    from model.gp_ensemble import GPEnsemble

    n = 25
    d = 2
    rng = np.random.default_rng(321)
    train_x = rng.uniform(size=(n, d))
    train_y = rng.normal(size=(n, 1))

    model = GPEnsemble(train_x, train_y)

    xq = rng.uniform(size=(3, d))
    mvn_incl = model.posterior(xq, k=5, exclude_nearest=False)
    mvn_excl = model.posterior(xq, k=5, exclude_nearest=True)

    assert mvn_incl.mu.shape == (3, 1)
    assert mvn_excl.mu.shape == (3, 1)
    assert mvn_incl.se.shape == (3, 1)
    assert mvn_excl.se.shape == (3, 1)

    assert np.all(mvn_excl.se >= mvn_incl.se - 1e-12)


def test_gp_ensemble_posterior_batch_shapes():
    import numpy as np

    from model.gp_ensemble import GPEnsemble

    n = 40
    d = 5
    rng = np.random.default_rng(42)
    train_x = rng.uniform(size=(n, d))
    train_y = rng.normal(size=(n, 1))

    model = GPEnsemble(train_x, train_y)

    b = 10
    xq = rng.uniform(size=(b, d))
    mvn = model.posterior(xq, k=7)

    assert mvn.mu.shape == (b, 1)
    assert mvn.se.shape == (b, 1)
    assert np.all(np.isfinite(mvn.mu))
    assert np.all(np.isfinite(mvn.se))


def test_gp_ensemble_exclude_nearest_requires_at_least_two_neighbors_n_equals_1():
    import numpy as np
    import pytest

    from model.gp_ensemble import GPEnsemble

    n = 1
    d = 2
    rng = np.random.default_rng(7)
    train_x = rng.uniform(size=(n, d))
    train_y = rng.normal(size=(n, 1))

    model = GPEnsemble(train_x, train_y)

    xq = train_x.copy()
    with pytest.raises(AssertionError):
        model.posterior(xq, k=1, exclude_nearest=True)


def test_gp_ensemble_exclude_nearest_k_equals_1_should_error_cleanly():
    import numpy as np
    import pytest

    from model.gp_ensemble import GPEnsemble

    n = 10
    d = 3
    rng = np.random.default_rng(9)
    train_x = rng.uniform(size=(n, d))
    train_y = rng.normal(size=(n, 1))

    model = GPEnsemble(train_x, train_y)

    xq = rng.uniform(size=(2, d))
    with pytest.raises(AssertionError):
        model.posterior(xq, k=1, exclude_nearest=True)


def test_gp_ensemble_num_gps_se_monotonic_decrease():
    import numpy as np

    from model.gp_ensemble import GPEnsemble

    n = 60
    d = 4
    rng = np.random.default_rng(12345)
    train_x = rng.uniform(size=(n, d))
    train_y = rng.normal(size=(n, 1))

    xq = rng.uniform(size=(5, d))

    m1 = GPEnsemble(train_x, train_y, num_gps=1)
    m5 = GPEnsemble(train_x, train_y, num_gps=5)

    mvn1 = m1.posterior(xq, k=8, exclude_nearest=False)
    mvn5 = m5.posterior(xq, k=8, exclude_nearest=False)

    assert mvn1.mu.shape == (5, 1)
    assert mvn1.se.shape == (5, 1)
    assert mvn5.mu.shape == (5, 1)
    assert mvn5.se.shape == (5, 1)

    assert np.all(mvn5.se <= mvn1.se + 1e-12)


def test_gp_ensemble_num_gps_type_validation_rejects_numpy_int():
    import numpy as np
    import pytest

    from model.gp_ensemble import GPEnsemble

    n = 10
    d = 3
    rng = np.random.default_rng(7)
    train_x = rng.uniform(size=(n, d))
    train_y = rng.normal(size=(n, 1))

    with pytest.raises(AssertionError):
        GPEnsemble(train_x, train_y, num_gps=np.int64(3))


def test_gp_ensemble_num_gps_exclude_nearest_and_variance():
    import numpy as np

    from model.gp_ensemble import GPEnsemble

    n = 50
    d = 2
    rng = np.random.default_rng(202)
    train_x = rng.uniform(size=(n, d))
    train_y = rng.normal(size=(n, 1))

    xq = rng.uniform(size=(6, d))

    m1 = GPEnsemble(train_x, train_y, num_gps=1)
    m4 = GPEnsemble(train_x, train_y, num_gps=4)

    mvn1_inc = m1.posterior(xq, k=6, exclude_nearest=False)
    mvn1_exc = m1.posterior(xq, k=6, exclude_nearest=True)

    mvn4_inc = m4.posterior(xq, k=6, exclude_nearest=False)
    mvn4_exc = m4.posterior(xq, k=6, exclude_nearest=True)

    assert np.all(mvn1_exc.se >= mvn1_inc.se - 1e-12)
    assert np.all(mvn4_exc.se >= mvn4_inc.se - 1e-12)
    assert np.all(mvn4_inc.se <= mvn1_inc.se + 1e-12)
    assert np.all(mvn4_exc.se <= mvn1_exc.se + 1e-12)


def test_gp_ensemble_num_gps_interpolates_at_train_point_with_zero_variance():
    import numpy as np

    from model.gp_ensemble import GPEnsemble

    n = 1
    d = 3
    rng = np.random.default_rng(55)
    train_x = rng.uniform(size=(n, d))
    train_y = rng.normal(size=(n, 1))

    model = GPEnsemble(train_x, train_y, num_gps=3)

    mvn = model.posterior(train_x.copy(), k=1, exclude_nearest=False)
    assert mvn.mu.shape == (1, 1)
    assert mvn.se.shape == (1, 1)
    assert np.allclose(mvn.mu, train_y, atol=1e-10)
    assert float(mvn.se) == 0.0
