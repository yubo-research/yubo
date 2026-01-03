import numpy as np


def test_enn_per_dimension_weighting_changes_neighbor_identity():
    from model.enn_weighter import ENNWeighter

    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rng = np.random.default_rng(0)
    filler = rng.uniform(5.0, 6.0, size=(200, 3)).astype(np.float32)
    x = np.vstack([base, filler]).astype(np.float32)
    q = np.array([[0.6, 0.6, 0.0]], dtype=np.float32)

    y_dim2 = x[:, 1:2]
    enn_dim2 = ENNWeighter(k=1, small_world_M=None, weighting="sobol_indices")
    enn_dim2.add(x, y_dim2)
    w2 = enn_dim2.weights
    idx2, _ = enn_dim2._enn.about_neighbors(q * w2, k=1)
    assert idx2.shape == (1, 1)
    assert idx2[0, 0] == 2

    y_dim1 = x[:, 0:1]
    enn_dim1 = ENNWeighter(k=1, small_world_M=None, weighting="sobol_indices")
    enn_dim1.add(x, y_dim1)
    w1 = enn_dim1.weights
    idx1, _ = enn_dim1._enn.about_neighbors(q * w1, k=1)
    assert idx1.shape == (1, 1)
    assert idx1[0, 0] == 1
