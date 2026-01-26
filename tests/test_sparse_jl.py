import numpy as np
from scipy.stats import spearmanr

from sampling.sparse_jl import block_sparse_jl_transform


def test_sparse_jl_preserves_neighbors_and_correlations():
    N = 100
    D = 1000
    R = D // 10
    C = 30
    eps = 0.2
    s = 4
    d = int(8 * np.log(C + 1) / (eps**2) + 0.5)
    assert d < D
    assert s < d

    theta_base = np.linspace(0.1, 1.0, D)
    rng0 = np.random.default_rng(2024)
    theta0 = np.stack(
        [theta_base + rng0.normal(scale=0.01, size=D) for _ in range(N)], axis=0
    )
    y0 = np.stack(
        [block_sparse_jl_transform(theta0[i], d, s=s) for i in range(N)], axis=0
    )
    rng = np.random.default_rng(123)
    dists_orig = np.zeros((C, N))
    dists_embed = np.zeros((C, N))
    pearson_all = np.zeros(C)
    spearman_all = np.zeros(C)
    for c in range(C):
        indices = rng.choice(D, size=R, replace=False)
        delta = np.zeros(D)
        delta[indices] = rng.normal(scale=0.05, size=R)
        theta_c = theta0[0] + delta
        y_c = block_sparse_jl_transform(theta_c, d, s=s)
        dists_orig[c] = np.linalg.norm(theta0 - theta_c, axis=1)
        dists_embed[c] = np.linalg.norm(y0 - y_c, axis=1)
        pearson_all[c] = np.corrcoef(dists_orig[c], dists_embed[c])[0, 1]
        spearman_all[c] = spearmanr(dists_orig[c], dists_embed[c])[0]

    nn_orig = np.argmin(dists_orig, axis=1)
    nn_embed = np.argmin(dists_embed, axis=1)
    nn_match_frac = np.mean(nn_orig == nn_embed)
    assert nn_match_frac > 0.9
    assert float(np.mean(pearson_all)) > 0.6
    assert float(np.median(pearson_all)) > 0.6
    assert float(np.mean(spearman_all)) > 0.6
    assert float(np.median(spearman_all)) > 0.6


def test_sparse_jl_without_replacement_per_column():
    D = 32
    d = 16
    s = 8
    seed = 0
    j = 0
    x = np.zeros(D)
    x[j] = 1.0
    y = block_sparse_jl_transform(x, d=d, s=s, seed=seed)
    idx = np.nonzero(np.abs(y) > 1e-12)[0]
    assert len(idx) == s
    assert np.allclose(np.abs(y[idx]), 1.0 / np.sqrt(s))


def test_sparse_jl_zero_maps_to_zero():
    D = 50
    d = 12
    s = 4
    x = np.zeros(D)
    y = block_sparse_jl_transform(x, d=d, s=s, seed=7)
    assert np.count_nonzero(y) == 0
    assert np.allclose(y, 0.0)


def test_sparse_jl_linearity_and_determinism():
    D = 64
    d = 16
    s = 8
    rng = np.random.default_rng(2025)
    x = rng.normal(size=D)
    z = rng.normal(size=D)
    seed = 123
    y_sum = block_sparse_jl_transform(x + z, d=d, s=s, seed=seed)
    y_x = block_sparse_jl_transform(x, d=d, s=s, seed=seed)
    y_z = block_sparse_jl_transform(z, d=d, s=s, seed=seed)
    assert np.allclose(y_sum, y_x + y_z)
    y1 = block_sparse_jl_transform(x, d=d, s=s, seed=seed)
    y2 = block_sparse_jl_transform(x, d=d, s=s, seed=seed)
    assert np.array_equal(y1, y2)
    y3 = block_sparse_jl_transform(x, d=d, s=s, seed=seed + 1)
    assert not np.array_equal(y1, y3)


def test_sparse_jl_raises_when_s_exceeds_d():
    D = 10
    d = 4
    s = 5
    x = np.ones(D)
    import pytest

    with pytest.raises(ValueError):
        block_sparse_jl_transform(x, d=d, s=s, seed=0)


def test_sparse_jl_timing_prints():
    import time

    rng = np.random.default_rng(2026)
    configs = [(5000, 256, 4), (20000, 512, 4)]
    for D, d, s in configs:
        x = rng.normal(size=D)
        t0 = time.perf_counter()
        y = block_sparse_jl_transform(x, d=d, s=s, seed=0)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        print(
            f"D={D}, d={d}, s={s}, time_ms={elapsed_ms:.2f}, y_norm={float(np.linalg.norm(y)):.4f}"
        )


def test_sparse_jl_timing_sparse_x_prints():
    import time

    rng = np.random.default_rng(2027)
    D = 20000
    d = 512
    s = 4
    k = 100
    x = np.zeros(D)
    nz_idx = rng.choice(D, size=k, replace=False)
    x[nz_idx] = rng.normal(size=k)
    t0 = time.perf_counter()
    y = block_sparse_jl_transform(x, d=d, s=s, seed=1)
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0
    print(
        f"D={D}, d={d}, s={s}, k={k}, time_ms={elapsed_ms:.2f}, y_norm={float(np.linalg.norm(y)):.4f}"
    )
