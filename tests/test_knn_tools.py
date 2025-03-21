def test_random_directions():
    from sampling.knn_tools import random_directions

    u = random_directions(10, 3)
    assert u.shape == (10, 3)


def test_approx_ard():
    import numpy as np

    from sampling.knn_tools import approx_ard

    x_max = np.random.uniform(size=(1, 5))
    y_max = 1
    x_n = np.random.uniform(size=(9, 5))
    y_n = 1 - np.random.uniform(size=(9,))

    u = approx_ard(x_max, y_max, x_n, y_n)
    print(u)


def test_utils():
    import numpy as np

    from sampling.knn_tools import nearest_neighbor, random_directions

    from .test_enn import set_up_enn

    num_dim, n, train_x, train_y, k, enn = set_up_enn()

    u = random_directions(1, num_dim)
    assert u.shape == (1, num_dim)

    found_bdy = False
    found_x = False
    for _ in range(100):
        x = np.random.uniform(size=(1, num_dim))
        idx_nn, _ = nearest_neighbor(enn, x)
        if idx_nn == -1:
            found_bdy = True
        else:
            found_x = True
        if found_bdy and found_x:
            break
    else:
        assert False, (found_bdy, found_x)


def test_farthest_neighbor():
    import numpy as np

    from sampling.knn_tools import farthest_neighbor, random_directions

    from .test_enn import set_up_enn

    num_dim, n, train_x, train_y, k, enn = set_up_enn()

    # farthest_neighbor(enn, x_0: np.array, u: np.array, eps_bound: float = 1e-6):
    u = random_directions(1, num_dim)
    x_fn = farthest_neighbor(enn, train_x[[0]], u)
    assert np.all(enn.neighbors(x_fn, k=1) == train_x[[0]])


def test_farthest_neighbor_n():
    import numpy as np

    from sampling.knn_tools import farthest_neighbor, random_directions

    from .test_enn import set_up_enn

    num_dim, n, train_x, train_y, k, enn = set_up_enn()

    u = random_directions(3, num_dim)

    x_fn_1 = farthest_neighbor(enn, train_x[[1]], u[[1]])

    x_fn = farthest_neighbor(enn, train_x[:3], u)
    assert np.abs(x_fn[[1]] - x_fn_1).mean() < 1e-4
    for i in range(x_fn.shape[0]):
        assert np.all(enn.neighbors(x_fn[[i]], k=1) == train_x[[i]])
