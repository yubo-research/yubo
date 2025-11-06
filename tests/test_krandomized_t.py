def test_sub_k_shapes_and_values_when_query_equals_subset_torch():
    import numpy as np
    import torch

    from model.krandomized_t import KRandomizedT

    rng = np.random.default_rng(0)
    n, d = 40, 3
    train_x = torch.from_numpy(rng.uniform(size=(n, d)).astype(np.float64))
    kr = KRandomizedT(train_x)

    idxs = torch.tensor([0, 3, 5, 11, 17, 23, 39], dtype=torch.int64)
    Kxx, Kx = kr.sub_k(idxs, train_x.index_select(0, idxs))

    Kxx_np = Kxx.numpy()
    Kx_np = Kx.numpy()
    assert Kxx_np.shape == (len(idxs), len(idxs))
    assert np.allclose(Kxx_np, Kxx_np.T, atol=1e-10)
    assert np.all(np.isfinite(Kxx_np))
    assert np.allclose(np.diag(Kxx_np), 1.0, atol=1e-12)

    assert Kx_np.shape == (len(idxs), len(idxs))
    assert np.allclose(Kx_np, Kxx_np, atol=1e-10)


def test_sub_k_query_single_point_returns_vector_and_matches_row_of_Kxx_torch():
    import numpy as np
    import torch

    from model.krandomized_t import KRandomizedT

    rng = np.random.default_rng(1)
    n, d = 25, 4
    train_x = torch.from_numpy(rng.uniform(size=(n, d)).astype(np.float64))
    kr = KRandomizedT(train_x)

    idxs = torch.tensor([2, 6, 7, 11, 18], dtype=torch.int64)
    Kxx, _ = kr.sub_k(idxs, train_x.index_select(0, idxs))

    pos = 2
    j = idxs[pos]
    Kxx2, Kx_vec = kr.sub_k(idxs, train_x[int(j)])

    Kxx_np = Kxx.numpy()
    Kxx2_np = Kxx2.numpy()
    Kx_vec_np = Kx_vec.numpy()

    assert Kxx2_np.shape == Kxx_np.shape
    assert np.allclose(Kxx2_np, Kxx_np, atol=1e-10)
    assert Kx_vec_np.shape == (len(idxs),)
    assert np.allclose(Kx_vec_np, Kxx_np[pos], atol=1e-10)
    assert np.isclose(Kx_vec_np[pos], 1.0, atol=1e-12)


def test_sub_k_batched_vs_loop_consistency_torch():
    import numpy as np
    import torch

    from model.krandomized_t import KRandomizedT

    rng = np.random.default_rng(2)
    n, d = 50, 2
    train_x = torch.from_numpy(rng.uniform(size=(n, d)).astype(np.float64))
    kr = KRandomizedT(train_x)

    idxs = torch.tensor([0, 10, 20, 30, 40], dtype=torch.int64)
    b = 7
    Xq = torch.from_numpy(rng.uniform(size=(b, d)).astype(np.float64))

    _, Kx_batched = kr.sub_k(idxs, Xq)

    rows = []
    for i in range(b):
        _, row = kr.sub_k(idxs, Xq[i])
        assert row.ndim == 1
        rows.append(row.numpy())
    Kx_loop = np.vstack(rows)

    assert np.allclose(Kx_batched.numpy(), Kx_loop, atol=1e-12)


def test_sub_k_psd_on_random_subset_torch():
    import numpy as np
    import torch

    from model.krandomized_t import KRandomizedT

    rng = np.random.default_rng(3)
    n, d = 35, 5
    train_x = torch.from_numpy(rng.uniform(size=(n, d)).astype(np.float64))
    kr = KRandomizedT(train_x)

    idxs_np = rng.choice(n, size=12, replace=False)
    idxs = torch.from_numpy(idxs_np.astype(np.int64))
    Kxx, _ = kr.sub_k(idxs, train_x.index_select(0, idxs[:3]))

    w = torch.linalg.eigvalsh(Kxx).numpy()
    assert np.min(w) >= -1e-8


def test_sub_k_invalid_inputs_raise_torch():
    import numpy as np
    import torch

    from model.krandomized_t import KRandomizedT

    rng = np.random.default_rng(4)
    n, d = 10, 3
    train_x = torch.from_numpy(rng.uniform(size=(n, d)).astype(np.float64))
    kr = KRandomizedT(train_x)

    try:
        kr.sub_k(torch.tensor([[0, 1]], dtype=torch.int64), train_x[0])
        assert False
    except AssertionError:
        pass

    try:
        kr.sub_k(torch.tensor([0, 1], dtype=torch.int64), train_x[0][None, None, :])
        assert False
    except AssertionError:
        pass

    try:
        kr.sub_k(torch.tensor([0, 1], dtype=torch.int64), torch.ones((2, d + 1), dtype=torch.float64))
        assert False
    except AssertionError:
        pass

    try:
        kr.sub_k(torch.tensor([0, n + 1], dtype=torch.int64), train_x[0])
        assert False
    except IndexError:
        pass


def test_sub_k_float_indices_cast_and_work_torch():
    import numpy as np
    import torch

    from model.krandomized_t import KRandomizedT

    rng = np.random.default_rng(5)
    n, d = 12, 2
    train_x = torch.from_numpy(rng.uniform(size=(n, d)).astype(np.float64))
    kr = KRandomizedT(train_x)

    idxs_f = torch.tensor([0.0, 3.0, 7.0], dtype=torch.float64)
    idxs_i = torch.tensor([0, 3, 7], dtype=torch.int64)

    Kxx_i, Kx_i = kr.sub_k(idxs_i, train_x[1])

    try:
        Kxx_f, Kx_f = kr.sub_k(idxs_f, train_x[1])
        Kxx_f_np = Kxx_f.numpy()
        Kx_f_np = Kx_f.numpy()
        Kxx_i_np = Kxx_i.numpy()
        Kx_i_np = Kx_i.numpy()
        assert np.allclose(Kxx_f_np, Kxx_i_np, atol=1e-12)
        assert np.allclose(Kx_f_np, Kx_i_np, atol=1e-12)
    except AssertionError:
        assert False
