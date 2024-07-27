import numpy as np


def tight_bounding_box(X_0, X, num_keep, length_min=1e-9):
    x_0 = np.asarray(X_0)
    x = np.asarray(X)

    num_dim = len(x_0.flatten())

    def _bounds(length):
        bounds = np.zeros(shape=(2, num_dim))
        bounds[0, :] = np.maximum(0.0, x_0 - length)
        bounds[1, :] = np.minimum(1.0, x_0 + length)
        return bounds

    length_low = length_min
    length_high = np.sqrt(num_dim)

    while True:
        length_mid = (length_low + length_high) / 2
        bounds = _bounds(length_mid)
        num_in_box = np.sum(np.all((x > bounds[0, :]) & (x < bounds[1, :]), axis=1))
        if num_in_box < num_keep:
            length_low = length_mid
        elif num_in_box > num_keep:
            length_high = length_mid
        else:
            return bounds
