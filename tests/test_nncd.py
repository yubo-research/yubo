def test_api_and_feasibility():
    import numpy as np

    from sampling.nncd import nncd_weights

    np.random.seed(7)
    B, N, D = 4, 50, 7
    x = np.random.randn(B, N, D)
    y = np.random.randn(B, N, 1)

    w = nncd_weights(y=y, x=x, iters_per_dimension=1, eps=1e-9)

    assert isinstance(w, np.ndarray)
    assert w.shape == (B, D)
    assert w.dtype == np.float64
    s = w.sum(axis=1)
    assert np.allclose(s, 1.0, atol=1e-7)
    assert w.min() >= -1e-12
    assert w.max() <= 1.0 + 1e-12


def test_exact_recovery_on_synthetic():
    import numpy as np

    from sampling.nncd import nncd_weights

    np.random.seed(11)
    B, N, D = 3, 120, 5
    W_true = np.random.rand(B, D)
    W_true = W_true / W_true.sum(axis=1, keepdims=True)
    x = np.random.randn(B, N, D)
    y = np.einsum("bnd,bd->bn", x, W_true)[:, :, None]

    w = nncd_weights(y=y, x=x, iters_per_dimension=10, eps=1e-10)

    err = np.abs(w - W_true).max()
    assert err < 1e-3


def test_batching_equivalence():
    import numpy as np

    from sampling.nncd import nncd_weights

    np.random.seed(13)
    B, N, D = 1, 80, 6
    x1 = np.random.randn(B, N, D)
    y1 = np.random.randn(B, N, 1)

    w1 = nncd_weights(y=y1, x=x1, iters_per_dimension=3, eps=1e-8)

    x2 = np.concatenate([x1, x1], axis=0)
    y2 = np.concatenate([y1, y1], axis=0)
    w2 = nncd_weights(y=y2, x=x2, iters_per_dimension=3, eps=1e-8)

    assert w2.shape == (2, D)
    assert np.allclose(w2[0], w1[0], atol=1e-7)
    assert np.allclose(w2[1], w1[0], atol=1e-7)


def test_D_eq_1_returns_one():
    import numpy as np

    from sampling.nncd import nncd_weights

    np.random.seed(17)
    B, N, D = 5, 40, 1
    x = np.random.randn(B, N, D)
    y = np.random.randn(B, N, 1)

    w = nncd_weights(y=y, x=x, iters_per_dimension=2, eps=1e-9)

    assert w.shape == (B, D)
    assert np.allclose(w, 1.0, atol=1e-12)


def test_timing_large_case():
    import time

    import numpy as np

    from sampling.nncd import nncd_weights

    np.random.seed(23)
    B, N, D = 5, 100, 300
    x = np.random.randn(B, N, D)
    y = np.random.randn(B, N, 1)

    t0 = time.time()
    w = nncd_weights(y=y, x=x, iters_per_dimension=1, eps=1e-8)
    t1 = time.time()

    print("nncd_weights elapsed:", t1 - t0)
    assert w.shape == (B, D)
