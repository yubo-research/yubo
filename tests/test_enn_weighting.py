import numpy as np


def test_enn_per_dimension_weighting_changes_neighbor_identity():
    from model.enn import EpistemicNearestNeighbors

    x = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.zeros((4, 1), dtype=np.float32)
    q = np.array([[0.6, 0.6, 0.0]], dtype=np.float32)

    weights_dim2_heavy = np.array([1.0, 100.0, 1.0], dtype=np.float32)
    weights_dim1_heavy = np.array([100.0, 1.0, 1.0], dtype=np.float32)

    enn_dim2 = EpistemicNearestNeighbors(k=1, small_world_M=None, dim_weights=weights_dim2_heavy)
    enn_dim2.add(x, y)
    idx2, _ = enn_dim2.about_neighbors(q, k=1)
    assert idx2.shape == (1, 1)
    assert idx2[0, 0] == 2

    enn_dim1 = EpistemicNearestNeighbors(k=1, small_world_M=None, dim_weights=weights_dim1_heavy)
    enn_dim1.add(x, y)
    idx1, _ = enn_dim1.about_neighbors(q, k=1)
    assert idx1.shape == (1, 1)
    assert idx1[0, 0] == 1
