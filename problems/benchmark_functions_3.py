import numpy as np


class Zakharov:
    """
    See: https://www.sfu.ca/~ssurjano/zakharov.html
    """

    def __call__(self, x):
        x = 2.5 + 7.5 * np.array(x)
        s = np.sum(0.5 * np.arange(1, len(x) + 1) * x)
        return np.sum(x**2) + s**2 + s**4
