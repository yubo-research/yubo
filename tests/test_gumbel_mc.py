import numpy as np


def test_gumbel_monte_carlo():
    from sampling.sampling_util import gumbel

    np.random.seed(42)
    n = 10
    num_samples = 100000

    samples = np.random.normal(0, 1, (num_samples, n))
    max_samples = np.max(samples, axis=1)
    empirical_mean = np.mean(max_samples)

    theoretical_val = gumbel(n)

    relative_error = abs(empirical_mean - theoretical_val) / theoretical_val
    assert relative_error < 0.25


def test_gumbel_sweep():
    from sampling.sampling_util import gumbel

    np.random.seed(42)

    for n in range(1, 100, 3):
        gumbel_val = gumbel(n)
        print(f"n: {n}, gumbel_val: {gumbel_val}")
