def test_sub_k_shapes_and_values_when_query_equals_subset():
    import numpy as np

    from model.k_randomized import KRandomized

    rng = np.random.default_rng(0)
    n, d = 40, 3
    train_x = rng.uniform(size=(n, d))
    kr = KRandomized(train_x)

    idxs = np.array([0, 3, 5, 11, 17, 23, 39])
    Kxx, Kx = kr.sub_k(idxs, train_x[idxs])

    assert Kxx.shape == (len(idxs), len(idxs))
    assert np.allclose(Kxx, Kxx.T, atol=1e-10)
    assert np.all(np.isfinite(Kxx))
    assert np.allclose(np.diag(Kxx), 1.0, atol=1e-12)

    assert Kx.shape == (len(idxs), len(idxs))
    assert np.allclose(Kx, Kxx, atol=1e-10)


def test_sub_k_query_single_point_returns_vector_and_matches_row_of_Kxx():
    import numpy as np

    from model.k_randomized import KRandomized

    rng = np.random.default_rng(1)
    n, d = 25, 4
    train_x = rng.uniform(size=(n, d))
    kr = KRandomized(train_x)

    idxs = np.array([2, 6, 7, 11, 18])
    Kxx, _ = kr.sub_k(idxs, train_x[idxs])

    pos = 2
    j = idxs[pos]
    Kxx2, Kx_vec = kr.sub_k(idxs, train_x[j])

    assert Kxx2.shape == Kxx.shape
    assert np.allclose(Kxx2, Kxx, atol=1e-10)
    assert Kx_vec.shape == (len(idxs),)
    assert np.allclose(Kx_vec, Kxx[pos], atol=1e-10)
    assert np.isclose(Kx_vec[pos], 1.0, atol=1e-12)


def test_sub_k_batched_vs_loop_consistency():
    import numpy as np

    from model.k_randomized import KRandomized

    rng = np.random.default_rng(2)
    n, d = 50, 2
    train_x = rng.uniform(size=(n, d))
    kr = KRandomized(train_x)

    idxs = np.array([0, 10, 20, 30, 40])
    b = 7
    Xq = rng.uniform(size=(b, d))

    _, Kx_batched = kr.sub_k(idxs, Xq)

    rows = []
    for i in range(b):
        _, row = kr.sub_k(idxs, Xq[i])
        assert row.ndim == 1
        rows.append(row)
    Kx_loop = np.vstack(rows)

    assert np.allclose(Kx_batched, Kx_loop, atol=1e-12)


def test_sub_k_psd_on_random_subset():
    import numpy as np

    from model.k_randomized import KRandomized

    rng = np.random.default_rng(3)
    n, d = 35, 5
    train_x = rng.uniform(size=(n, d))
    kr = KRandomized(train_x)

    idxs = rng.choice(n, size=12, replace=False)
    Kxx, _ = kr.sub_k(idxs, train_x[idxs[:3]])

    w = np.linalg.eigvalsh(Kxx)
    assert np.min(w) >= -1e-8


def test_sub_k_invalid_inputs_raise():
    import numpy as np

    from model.k_randomized import KRandomized

    rng = np.random.default_rng(4)
    n, d = 10, 3
    train_x = rng.uniform(size=(n, d))
    kr = KRandomized(train_x)

    try:
        kr.sub_k(np.array([[0, 1]]), train_x[0])
        assert False
    except AssertionError:
        pass

    try:
        kr.sub_k(np.array([0, 1]), train_x[0][None, None, :])
        assert False
    except AssertionError:
        pass

    try:
        kr.sub_k(np.array([0, 1]), np.ones((2, d + 1)))
        assert False
    except AssertionError:
        pass

    try:
        kr.sub_k(np.array([0, n + 1]), train_x[0])
        assert False
    except IndexError:
        pass


def test_sub_k_float_indices_cast_and_work():
    import numpy as np

    from model.k_randomized import KRandomized

    rng = np.random.default_rng(5)
    n, d = 12, 2
    train_x = rng.uniform(size=(n, d))
    kr = KRandomized(train_x)

    idxs_f = np.array([0.0, 3.0, 7.0], dtype=float)
    idxs_i = np.array([0, 3, 7], dtype=int)

    Kxx_f, Kx_f = kr.sub_k(idxs_f, train_x[1])
    Kxx_i, Kx_i = kr.sub_k(idxs_i, train_x[1])

    assert np.allclose(Kxx_f, Kxx_i, atol=1e-12)
    assert np.allclose(Kx_f, Kx_i, atol=1e-12)
