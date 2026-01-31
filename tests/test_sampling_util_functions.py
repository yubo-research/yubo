import numpy as np
import torch


def test_greedy_maximin():
    from sampling.sampling_util import greedy_maximin

    np.random.seed(42)
    x = np.random.rand(100, 5)
    indices = greedy_maximin(x, 10)
    assert len(indices) == 10
    assert len(set(indices)) == 10


def test_top_k():
    from sampling.sampling_util import top_k

    x = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
    idx = top_k(x, 3)
    assert set(idx) == {1, 2, 4}


def test_top_k_larger_than_x():
    from sampling.sampling_util import top_k

    x = np.array([1.0, 5.0, 3.0])
    idx = top_k(x, 10)
    assert len(idx) == 3


def test_var_of_var_dev():
    from sampling.sampling_util import var_of_var_dev

    w = torch.tensor([0.25, 0.25, 0.25, 0.25])
    dev = torch.tensor([1.0, -1.0, 2.0, -2.0])
    result = var_of_var_dev(w, dev)
    assert torch.isfinite(result)


def test_qmc_normal_sample():
    from sampling.sampling_util import qmc_normal_sample

    mu = np.array([0.0, 0.0])
    cov = np.array([1.0, 1.0])
    samples = qmc_normal_sample(mu, cov, num_samples=10)
    assert samples.shape == (10, 2)


def test_raasp_np_choice():
    from sampling.sampling_util import raasp_np_choice

    x_center = np.array([0.5, 0.5, 0.5])
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 1.0])
    candidates = raasp_np_choice(x_center, lb, ub, 10)
    assert candidates.shape == (10, 3)
    assert np.all(candidates >= 0)
    assert np.all(candidates <= 1)


def test_raasp_np_p():
    from sampling.sampling_util import raasp_np_p

    np.random.seed(42)
    x_center = np.array([[0.5, 0.5, 0.5]])
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 1.0])
    candidates = raasp_np_p(x_center, lb, ub, 10)
    assert candidates.shape == (10, 3)


def test_raasp_np():
    from sampling.sampling_util import raasp_np

    np.random.seed(42)
    x_center = np.array([[0.5, 0.5, 0.5, 0.5, 0.5]])
    lb = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    candidates = raasp_np(x_center, lb, ub, 10, num_pert=2)
    assert candidates.shape == (10, 5)


def test_truncated_normal_np():
    from sampling.sampling_util import truncated_normal_np

    mu = np.array([[0.5, 0.5]])
    sigma = np.array([[0.1, 0.1]])
    lb = np.array([[0.0, 0.0]])
    ub = np.array([[1.0, 1.0]])
    samples = truncated_normal_np(mu, sigma, lb, ub, 100)
    assert samples.shape == (100, 2)
    assert np.all(samples >= 0)
    assert np.all(samples <= 1)


def test_sobol_perturb_np():
    from sampling.sampling_util import sobol_perturb_np

    np.random.seed(42)
    x_center = np.array([[0.5, 0.5, 0.5]])
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 1.0])
    mask = np.array([[True, False, True]] * 10)
    candidates = sobol_perturb_np(x_center, lb, ub, 10, mask)
    assert candidates.shape == (10, 3)


def test_raasp_torch():
    from sampling.sampling_util import raasp

    x_center = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64)
    lb = [0.0, 0.0, 0.0]
    ub = [1.0, 1.0, 1.0]
    candidates = raasp(x_center, lb, ub, 10, device="cpu", dtype=torch.float64)
    assert candidates.shape == (10, 3)


def test_raasp_turbo_np():
    from sampling.sampling_util import raasp_turbo_np

    np.random.seed(42)
    x_center = torch.tensor([[0.5, 0.5, 0.5]])
    lb = [0.0, 0.0, 0.0]
    ub = [1.0, 1.0, 1.0]
    candidates = raasp_turbo_np(x_center, lb, ub, 10, device="cpu", dtype=torch.float64)
    assert candidates.shape == (10, 3)
