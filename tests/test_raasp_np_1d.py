def _assert_basic_properties(x_candidates, num_candidates, num_dim, lb, ub):
    import numpy as np

    assert x_candidates.shape == (num_candidates, num_dim)
    assert np.all(x_candidates >= lb)
    assert np.all(x_candidates <= ub)


def _assert_single_dimension_perturbation(x_candidates, x_centers):
    import numpy as np

    num_candidates = len(x_candidates)
    num_centers = len(x_centers)

    for i in range(num_candidates):
        candidate = x_candidates[i]
        center_idx = i % num_centers
        center = x_centers[center_idx]

        diff_dims = np.where(np.abs(candidate - center) > 1e-10)[0]
        assert len(diff_dims) <= 1, f"Candidate {i} differs in {len(diff_dims)} dimensions"


def test_raasp_np_1d_basic_functionality():
    import numpy as np

    from sampling.sampling_util import raasp_np_1d

    np.random.seed(42)

    num_centers = 3
    num_dim = 4
    num_candidates = 10

    x_centers = np.random.uniform(0.2, 0.8, size=(num_centers, num_dim))
    lb = np.array([[0.0, 0.0, 0.0, 0.0]])
    ub = np.array([[1.0, 1.0, 1.0, 1.0]])

    x_candidates = raasp_np_1d(x_centers, lb, ub, num_candidates)
    _assert_basic_properties(x_candidates, num_candidates, num_dim, lb, ub)


def test_raasp_np_1d_shape_validation():
    import numpy as np

    from sampling.sampling_util import raasp_np_1d

    np.random.seed(42)

    x_centers = np.random.uniform(0.3, 0.7, size=(2, 3))
    lb = np.array([[0.1, 0.1, 0.1]])
    ub = np.array([[0.9, 0.9, 0.9]])

    assert x_centers.shape == (2, 3)
    assert lb.shape == (1, 3)
    assert ub.shape == (1, 3)

    x_candidates = raasp_np_1d(x_centers, lb, ub, 5)
    assert x_candidates.shape == (5, 3)


def test_raasp_np_1d_bounds_validation():
    import numpy as np

    from sampling.sampling_util import raasp_np_1d

    np.random.seed(42)

    x_centers = np.random.uniform(0.2, 0.8, size=(4, 2))
    lb = np.array([[0.1, 0.1]])
    ub = np.array([[0.9, 0.9]])

    assert np.all(x_centers >= lb)
    assert np.all(x_centers <= ub)

    x_candidates = raasp_np_1d(x_centers, lb, ub, 8)
    assert np.all(x_candidates >= lb)
    assert np.all(x_candidates <= ub)


def test_raasp_np_1d_single_dimension_perturbation():
    import numpy as np

    from sampling.sampling_util import raasp_np_1d

    np.random.seed(42)

    x_centers = np.array([[0.5, 0.5, 0.5], [0.3, 0.7, 0.4]])
    lb = np.array([[0.0, 0.0, 0.0]])
    ub = np.array([[1.0, 1.0, 1.0]])

    x_candidates = raasp_np_1d(x_centers, lb, ub, 6)
    _assert_single_dimension_perturbation(x_candidates, x_centers)


def test_raasp_np_1d_edge_cases():
    import numpy as np

    from sampling.sampling_util import raasp_np_1d

    np.random.seed(42)

    x_centers = np.array([[0.5]])
    lb = np.array([[0.0]])
    ub = np.array([[1.0]])

    x_candidates = raasp_np_1d(x_centers, lb, ub, 1)
    _assert_basic_properties(x_candidates, 1, 1, lb, ub)


def test_raasp_np_1d_high_dimensions():
    import numpy as np

    from sampling.sampling_util import raasp_np_1d

    np.random.seed(42)

    num_centers = 3
    num_dim = 10
    num_candidates = 15

    x_centers = np.random.uniform(0.1, 0.9, size=(num_centers, num_dim))
    lb = np.array([[0.0] * num_dim])
    ub = np.array([[1.0] * num_dim])

    x_candidates = raasp_np_1d(x_centers, lb, ub, num_candidates)
    _assert_basic_properties(x_candidates, num_candidates, num_dim, lb, ub)


def test_raasp_np_1d_tight_bounds():
    import numpy as np

    from sampling.sampling_util import raasp_np_1d

    np.random.seed(42)

    x_centers = np.array([[0.5, 0.5], [0.6, 0.4]])
    lb = np.array([[0.4, 0.3]])
    ub = np.array([[0.7, 0.6]])

    assert np.all(x_centers >= lb)
    assert np.all(x_centers <= ub)

    x_candidates = raasp_np_1d(x_centers, lb, ub, 5)
    assert np.all(x_candidates >= lb)
    assert np.all(x_candidates <= ub)


def test_raasp_np_1d_multiple_centers():
    import numpy as np

    from sampling.sampling_util import raasp_np_1d

    np.random.seed(42)

    num_centers = 5
    num_dim = 3
    num_candidates = 12

    x_centers = np.random.uniform(0.2, 0.8, size=(num_centers, num_dim))
    lb = np.array([[0.0, 0.0, 0.0]])
    ub = np.array([[1.0, 1.0, 1.0]])

    x_candidates = raasp_np_1d(x_centers, lb, ub, num_candidates)
    _assert_basic_properties(x_candidates, num_candidates, num_dim, lb, ub)
    _assert_single_dimension_perturbation(x_candidates, x_centers)
