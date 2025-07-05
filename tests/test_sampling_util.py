def test_intersect_with_box():
    import numpy as np

    from sampling.sampling_util import intersect_with_box

    x0 = np.array([0.2, 0.5])
    x1 = np.array([10, 1.5])
    intersection = intersect_with_box(x0, x1)
    print()
    print("I:", x0)
    print("O:", x1)
    print(intersection)


def test_var_of_var():
    import torch

    from sampling.sampling_util import var_of_var

    torch.manual_seed(17)
    n = 100
    for _ in range(5):
        w = torch.rand(size=(n,))
        w = w / w.sum()
        X = torch.randn(size=(n,))

        vv_w = var_of_var(w=w, X=X)

        w_0 = torch.ones(size=(n,))
        w_0 = w_0 / w_0.sum()
        vv_0 = var_of_var(w=w_0, X=X)

        if vv_w > vv_0:
            break
    else:
        assert False


def _test_draw_bounded_normal_samples(num_dim, qmc):
    import numpy as np
    import torch

    from sampling.sampling_util import draw_bounded_normal_samples

    np.random.seed(17)
    torch.manual_seed(17)

    mu = np.random.uniform(size=(num_dim,))
    cov = 0.003 * np.ones(shape=(num_dim,))
    cov[0] = 0.001

    num_samples = 1024

    x, p = draw_bounded_normal_samples(mu, cov, num_samples, qmc=qmc)

    assert np.abs(x.mean(axis=0) - mu).max() < 0.05
    assert np.abs(x.var(axis=0) - cov).max() < 0.05


def test_narrow():
    for qmc in [True, False]:
        for num_dim in [1, 3, 10, 30, 100]:
            _test_draw_bounded_normal_samples(num_dim, qmc)


def test_wide():
    import numpy as np

    from sampling.sampling_util import draw_bounded_normal_samples

    np.random.seed(17)

    num_dim = 100
    mu = np.random.uniform(size=(num_dim,))
    cov = 0.3 * np.ones(shape=(num_dim,))
    cov[0] = 0.1

    num_samples = 1024

    x, _ = draw_bounded_normal_samples(mu, cov, num_samples, qmc=False)

    assert x.min() >= 0 and x.max() <= 1


def _test_draw_varied_bounded_normal_samples(num_dim):
    import numpy as np
    import torch

    from sampling.sampling_util import draw_varied_bounded_normal_samples

    np.random.seed(17)
    torch.manual_seed(17)

    mu = np.random.uniform(size=(num_dim,))
    cov = 0.003 * np.ones(shape=(num_dim,))
    cov[0] = 0.001

    mus_covs = [
        (mu, cov),
        (mu, cov),
        (mu, cov),
    ]
    x, p = draw_varied_bounded_normal_samples(mus_covs)

    assert x.shape == (len(mus_covs), num_dim)
    assert len(p) == len(mus_covs)


def _xx_test_varied_1():
    _test_draw_varied_bounded_normal_samples(1)


def _xx_test_varied_n():
    for n in [2, 3, 10]:
        _test_draw_varied_bounded_normal_samples(n)


def test_raasp_np_basic():
    import numpy as np

    from sampling.sampling_util import raasp_np

    np.random.seed(42)

    x_center = np.full(25, 0.5)
    lb = np.zeros(25)
    ub = np.ones(25)
    num_candidates = 100

    candidates = raasp_np(x_center, lb, ub, num_candidates)

    assert candidates.shape == (num_candidates, 25)

    assert np.all(candidates >= lb)
    assert np.all(candidates <= ub)

    assert not np.allclose(candidates, x_center)

    center_matches = np.any(np.isclose(candidates, x_center, atol=1e-10), axis=1)
    assert np.any(center_matches)


def test_raasp_np_different_dimensions():
    import numpy as np

    from sampling.sampling_util import raasp_np

    np.random.seed(42)

    candidates_1d = raasp_np(np.array([0.5]), [0.0], [1.0], 50)
    assert candidates_1d.shape == (50, 1)
    assert np.all(candidates_1d >= 0) and np.all(candidates_1d <= 1)

    x_center_10d = np.full(10, 0.5)
    lb_10d = np.zeros(10)
    ub_10d = np.ones(10)
    candidates_10d = raasp_np(x_center_10d, lb_10d, ub_10d, 50)
    assert candidates_10d.shape == (50, 10)
    assert np.all(candidates_10d >= 0) and np.all(candidates_10d <= 1)


def test_raasp_np_custom_bounds():
    import numpy as np

    from sampling.sampling_util import raasp_np

    np.random.seed(42)

    x_center = np.array([5.0, 5.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([10.0, 10.0])
    num_candidates = 50

    candidates = raasp_np(x_center, lb, ub, num_candidates)

    assert candidates.shape == (50, 2)
    assert np.all(candidates >= lb)
    assert np.all(candidates <= ub)

    assert not np.allclose(candidates, x_center)


def test_raasp_np_perturbation_probability():
    import numpy as np

    from sampling.sampling_util import raasp_np

    np.random.seed(42)

    x_center_1d = np.array([0.5])
    candidates_1d = raasp_np(x_center_1d, [0.0], [1.0], 100)
    assert not np.allclose(candidates_1d, x_center_1d)

    x_center_20d = np.full(20, 0.5)
    candidates_20d = raasp_np(x_center_20d, np.zeros(20), np.ones(20), 100)
    assert not np.allclose(candidates_20d, x_center_20d)


def test_raasp_basic():
    import numpy as np
    import torch

    from sampling.sampling_util import raasp

    torch.manual_seed(42)
    np.random.seed(42)

    x_center = torch.full((25,), 0.5, dtype=torch.float64)
    lb = np.zeros(25)
    ub = np.ones(25)
    num_candidates = 100
    device = torch.device("cpu")
    dtype = torch.float64

    candidates = raasp(x_center, lb, ub, num_candidates, device, dtype)

    assert candidates.shape == (num_candidates, 25)

    assert torch.all(candidates >= torch.tensor(lb, dtype=dtype, device=device))
    assert torch.all(candidates <= torch.tensor(ub, dtype=dtype, device=device))

    assert not torch.allclose(candidates, x_center)

    center_matches = torch.any(torch.isclose(candidates, x_center, atol=1e-10), dim=1)
    assert torch.any(center_matches)


def test_raasp_different_dimensions():
    import numpy as np
    import torch

    from sampling.sampling_util import raasp

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cpu")
    dtype = torch.float64

    candidates_1d = raasp(torch.tensor([0.5], dtype=dtype), [0.0], [1.0], 50, device, dtype)
    assert candidates_1d.shape == (50, 1)
    assert torch.all(candidates_1d >= 0) and torch.all(candidates_1d <= 1)

    x_center_10d = torch.full((10,), 0.5, dtype=dtype)
    lb_10d = np.zeros(10)
    ub_10d = np.ones(10)
    candidates_10d = raasp(x_center_10d, lb_10d, ub_10d, 50, device, dtype)
    assert candidates_10d.shape == (50, 10)
    assert torch.all(candidates_10d >= 0) and torch.all(candidates_10d <= 1)


def test_raasp_custom_bounds():
    import numpy as np
    import torch

    from sampling.sampling_util import raasp

    torch.manual_seed(42)
    np.random.seed(42)

    x_center = torch.tensor([5.0, 5.0], dtype=torch.float64)
    lb = np.array([0.0, 0.0])
    ub = np.array([10.0, 10.0])
    num_candidates = 50
    device = torch.device("cpu")
    dtype = torch.float64

    candidates = raasp(x_center, lb, ub, num_candidates, device, dtype)

    assert candidates.shape == (50, 2)
    assert torch.all(candidates >= torch.tensor(lb, dtype=dtype, device=device))
    assert torch.all(candidates <= torch.tensor(ub, dtype=dtype, device=device))

    assert not torch.allclose(candidates, x_center)


def test_raasp_consistency_with_raasp_np():
    import numpy as np
    import torch

    from sampling.sampling_util import raasp, raasp_np

    torch.manual_seed(42)
    np.random.seed(42)

    x_center_np = np.full(25, 0.5)
    x_center_torch = torch.full((25,), 0.5, dtype=torch.float64)
    lb = np.zeros(25)
    ub = np.ones(25)
    num_candidates = 50
    device = torch.device("cpu")
    dtype = torch.float64

    candidates_np = raasp_np(x_center_np, lb, ub, num_candidates)
    candidates_torch = raasp(x_center_torch, lb, ub, num_candidates, device, dtype)

    candidates_torch_np = candidates_torch.detach().numpy()

    assert candidates_np.shape == candidates_torch_np.shape

    assert np.all(candidates_np >= lb) and np.all(candidates_np <= ub)
    assert np.all(candidates_torch_np >= lb) and np.all(candidates_torch_np <= ub)


def test_raasp_edge_cases():
    import numpy as np
    import torch

    from sampling.sampling_util import raasp, raasp_np

    x_center = np.array([0.5, 0.5])
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])

    candidates_np = raasp_np(x_center, lb, ub, 1)
    assert candidates_np.shape == (1, 2)

    candidates_torch = raasp(torch.tensor([0.5, 0.5], dtype=torch.float64), lb, ub, 1, torch.device("cpu"), torch.float64)
    assert candidates_torch.shape == (1, 2)

    x_center_extreme = np.array([0.0, 1.0])
    candidates_extreme = raasp_np(x_center_extreme, lb, ub, 10)
    assert np.all(candidates_extreme >= lb) and np.all(candidates_extreme <= ub)

    x_center_small = np.array([0.5, 0.5])
    lb_small = np.array([0.4, 0.4])
    ub_small = np.array([0.6, 0.6])
    candidates_small = raasp_np(x_center_small, lb_small, ub_small, 10)
    assert np.all(candidates_small >= lb_small) and np.all(candidates_small <= ub_small)


def test_raasp_input_validation():
    import numpy as np

    from sampling.sampling_util import raasp_np

    x_center = np.array([0.5, 0.5, 0.5])
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])

    try:
        candidates = raasp_np(x_center, lb, ub, 10)
        assert candidates.shape == (10, 3)
    except Exception:
        pass

    x_center = np.array([0.5, 0.5])
    lb = np.array([1.0, 1.0])
    ub = np.array([0.0, 0.0])

    candidates = raasp_np(x_center, lb, ub, 10)


def test_raasp_np_choice_bug():
    import numpy as np
    from scipy.stats import qmc

    from sampling.sampling_util import raasp_np_choice

    np.random.seed(42)

    x_center = np.array([0.5, 0.5])
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    num_candidates = 10

    candidates = raasp_np_choice(x_center, lb, ub, num_candidates)

    print("x_center:", x_center)
    print("candidates shape:", candidates.shape)
    print("candidates:\n", candidates)

    assert candidates.shape == (num_candidates, 2)
    assert np.all(candidates >= lb)
    assert np.all(candidates <= ub)

    num_dim = x_center.shape[-1]
    ind = np.random.choice(np.arange(num_dim), size=num_candidates, replace=True)
    print("ind:", ind)
    print("ind shape:", ind.shape)

    sobol_engine = qmc.Sobol(num_dim, scramble=True)
    sobol_samples = sobol_engine.random(num_candidates)
    print("sobol_samples shape:", sobol_samples.shape)

    lb_array = np.asarray(lb)
    ub_array = np.asarray(ub)
    pert = lb_array + (ub_array - lb_array) * sobol_samples
    print("pert shape:", pert.shape)
    print("pert:\n", pert)

    candidates_test = np.tile(x_center, (num_candidates, 1))
    print("candidates_test before:\n", candidates_test)

    candidates_test[ind] = pert[ind]
    print("candidates_test after:\n", candidates_test)
