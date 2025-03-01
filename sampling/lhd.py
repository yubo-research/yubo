import numpy as np


def latin_hypercube_design(num_samples, num_dim):
    # From turbo.utils
    X = np.zeros((num_samples, num_dim))
    centers = (1.0 + 2.0 * np.arange(0.0, num_samples)) / float(2 * num_samples)
    for i in range(num_dim):
        X[:, i] = centers[np.random.permutation(num_samples)]

    # Add some perturbations within each box
    X += np.random.uniform(-1.0, 1.0, (num_samples, num_dim)) / float(2 * num_samples)
    return X
