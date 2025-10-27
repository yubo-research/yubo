def test_small_world_hnsw_self_is_nearest_neighbor():
    import numpy as np

    from model.enn import EpistemicNearestNeighbors

    num_dim = 5
    n = 100
    x = np.random.uniform(size=(n, num_dim)).astype(np.float32)
    y = np.random.normal(size=(n, 1)).astype(np.float32)

    enn_hnsw = EpistemicNearestNeighbors(k=3, small_world_M=16)
    enn_hnsw.add(x, y)

    q = x[[0, 10, 20]]
    idx_h, dist2_h = enn_hnsw.about_neighbors(q, k=1)

    assert idx_h.shape == (3, 1)
    assert np.all(idx_h.flatten() == np.array([0, 10, 20]))
    assert np.allclose(dist2_h, 0.0)


def test_small_world_hnsw_matches_flat_on_train_queries():
    import numpy as np

    from model.enn import EpistemicNearestNeighbors

    num_dim = 6
    n = 200
    x = np.random.uniform(size=(n, num_dim)).astype(np.float32)
    y = np.random.normal(size=(n, 1)).astype(np.float32)

    enn_flat = EpistemicNearestNeighbors(k=3, small_world_M=None)
    enn_flat.add(x, y)

    enn_hnsw = EpistemicNearestNeighbors(k=3, small_world_M=16)
    enn_hnsw.add(x, y)

    q_idx = np.random.choice(np.arange(n), size=5, replace=False)
    q = x[q_idx]

    idx_f, dist2_f = enn_flat.about_neighbors(q, k=3)
    idx_h, dist2_h = enn_hnsw.about_neighbors(q, k=3)

    assert idx_f.shape == idx_h.shape
    assert dist2_f.shape == dist2_h.shape
    assert np.all(idx_f[:, 0] == idx_h[:, 0])
