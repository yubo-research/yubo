import numpy as np


def tight_bounding_box_1(X_0, X, num_keep, length_min=1e-9, max_iterations=100):
    x_0 = np.asarray(X_0)
    x = np.asarray(X)

    num_dim = len(x_0.flatten())

    def _bounds(length):
        eps = 1e-9
        bounds = np.zeros(shape=(2, num_dim))
        bounds[0, :] = np.maximum(0.0 - eps, x_0 - length)
        bounds[1, :] = np.minimum(1.0 + eps, x_0 + length)
        return bounds

    if len(X) <= num_keep:
        return _bounds(100)

    length_low = length_min
    length_high = np.sqrt(num_dim)

    for _ in range(max_iterations):
        length_mid = (length_low + length_high) / 2
        bounds = _bounds(length_mid)
        num_in_box = np.sum(np.all((x > bounds[0, :]) & (x < bounds[1, :]), axis=1))
        # print("N:", num_in_box, length_mid, bounds)
        if num_in_box < num_keep:
            length_low = length_mid
        elif num_in_box > num_keep:
            length_high = length_mid
        else:
            return bounds
    assert False, ("Failed to find lengthscale", num_keep, X_0, X)


def tight_bounding_box(X_0, X, num_keep, length_min=1e-9, delta_length=1e-6, max_iterations=100, eps_bounds=1e-9):
    x_0 = np.asarray(X_0)
    x = np.asarray(X)
    num_dim = len(x_0.flatten())

    def _bounds(length):
        bounds = np.zeros(shape=(2, num_dim))
        bounds[0, :] = np.maximum(0.0, x_0 - length)
        bounds[1, :] = np.minimum(1.0, x_0 + length)
        return bounds

    if len(X) <= num_keep:
        return np.arange(len(X)), _bounds(100)

    dist = np.sqrt(((x - x_0) ** 2).sum(axis=1))
    idx = np.argpartition(dist, num_keep)[:num_keep]
    x = x[idx, :]

    assert x_0 in x, (num_keep, x_0, x, idx)

    length_low = length_min
    length_high = np.sqrt(num_dim)

    for i_iter in range(max_iterations):
        assert length_high > length_low, (i_iter, length_high, length_low)
        length_mid = (length_low + length_high) / 2
        bounds = _bounds(length_mid)
        num_in_box = np.sum(np.all((x > bounds[0, :] - eps_bounds) & (x < bounds[1, :] + eps_bounds), axis=1))

        if num_in_box == num_keep:
            length_high = length_mid
        elif num_in_box < num_keep:
            length_low = length_mid
        else:
            assert False, ("Impossible", num_in_box, num_keep)

        if length_high - length_low < delta_length:
            return idx, _bounds(length_high)

    assert False, ("Failed to find lengthscale", num_keep, X_0, X)
