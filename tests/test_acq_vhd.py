def test_find_max_distance():
    # import numpy as np

    from acq.acq_vhd import find_farthest_neighbor

    from .test_knn_tools import set_up_enn

    num_dim, n, train_x, train_y, k, enn = set_up_enn()
    x_0 = train_x[[0]]
    x = find_farthest_neighbor(enn, x_0)
    print(x)
