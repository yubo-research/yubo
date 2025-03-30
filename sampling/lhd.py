import numpy as np


# From turbo.utils
def latin_hypercube_design(num_samples, num_dim, seed=None):
    rng = np.random.default_rng(seed)

    X = np.zeros((num_samples, num_dim))
    centers = (1.0 + 2.0 * np.arange(0.0, num_samples)) / float(2 * num_samples)
    for i in range(num_dim):
        X[:, i] = centers[rng.permutation(num_samples)]

    X += rng.uniform(-1.0, 1.0, (num_samples, num_dim)) / float(2 * num_samples)
    return X
