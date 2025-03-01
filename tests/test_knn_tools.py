def _set_up_enn():
    import numpy as np

    from model.enn import EpsitemicNearestNeighbors

    num_dim = 5
    n = 10
    train_x = np.random.uniform(size=(n, num_dim))
    train_y = np.random.normal(size=(n, 1))

    k = 3

    enn = EpsitemicNearestNeighbors(train_x, train_y, k=k)
    return num_dim, n, train_x, train_y, k, enn


def test_utils():
    import numpy as np

    from sampling.knn_tools import _idx_nearest_neighbor, random_direction

    num_dim, n, train_x, train_y, k, enn = _set_up_enn()

    u = random_direction(num_dim)
    assert len(u) == num_dim

    found_bdy = False
    found_x = False
    for _ in range(100):
        x = np.random.uniform(size=(1, num_dim))
        idx_nn = _idx_nearest_neighbor(enn, x)
        if idx_nn == -1:
            found_bdy = True
        else:
            found_x = True
        if found_bdy and found_x:
            break
    else:
        assert False, (found_bdy, found_x)


def test_farthest_neighbor():
    # import numpy as np

    from sampling.knn_tools import farthest_neighbor, random_direction

    num_dim, n, train_x, train_y, k, enn = _set_up_enn()
    # x = np.random.uniform(size=(1, num_dim))

    # farthest_neighbor(enn, x_0: np.array, u: np.array, eps_bound: float = 1e-6):
    u = random_direction(num_dim)
    farthest_neighbor(enn, train_x[[0]], u)
