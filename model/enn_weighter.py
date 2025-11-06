import numpy as np

from model.enn import EpistemicNearestNeighbors
from sampling.sobol_indices import calculate_sobol_indices_np
from sampling.x_cov import evec_1


class ENNWeighter:
    def __init__(self, *, weighting: str, k=3, small_world_M=None):
        self._weighting = weighting
        self._x_center = None

        self._enn = EpistemicNearestNeighbors(k=k, small_world_M=small_world_M)
        self._weights = None
        self._xy = None

    def __len__(self):
        return len(self._enn)

    @property
    def weights(self):
        self._set_weights()
        return self._weights

    def _calc_weights(self, x, y):
        assert len(self) == 0
        if self._weighting == "sobol_indices":
            w = calculate_sobol_indices_np(x, y).astype(np.double)
            w = w / w.sum()
        elif self._weighting == "sigma_x":
            s = np.std(x, axis=0).astype(np.double)
            s = np.maximum(s, 1e-6)
            w = 1.0 / s
        elif self._weighting == "curvature":
            assert self._x_center is not None, "x_center is required for curvature weighting"
            w = calculate_curvature_weights_np(x, y, self._x_center)
        elif self._weighting == "sobol_over_sigma":
            s = np.std(x, axis=0).astype(np.double)
            s = np.maximum(s, 1e-6)
            w = calculate_sobol_indices_np(x, y).astype(np.double) / s
            w = w / w.sum()
        elif self._weighting == "sobol_over_evec":
            assert self._x_center is not None, "x_center is required for sobol_over_evec weighting"
            if x.shape[0] < 2:
                w = np.ones(x.shape[1])
            else:
                evec = np.maximum(1e-6, np.abs(evec_1(self._x_center, x)))
                w = calculate_sobol_indices_np(x, y).astype(np.double) / evec
            w = w / w.sum()
        else:
            assert False

        return np.maximum(1e-6, w).astype(np.double)

    def _set_weights(self):
        if self._weights is None:
            train_x, train_y = self._xy
            self._xy = "done"
            self._weights = self._calc_weights(train_x, train_y)
            self._enn.add(train_x * self._weights, train_y)

    def _rescale(self, x):
        self._set_weights()
        return x * self._weights

    def set_x_center(self, x_center):
        if x_center is None:
            return
        assert self._x_center is None, "x_center can only be set once"
        self._x_center = x_center.flatten()

    def add(self, x, y):
        assert self._xy is None, "You may only add once to an ENNWeighter"
        self._xy = (x, y)

    def __call__(self, x):
        return self.posterior(x)

    def posterior(self, x, *, k=None, exclude_nearest=False):
        return self._enn.posterior(self._rescale(x), k=k, exclude_nearest=exclude_nearest)


def calculate_curvature_weights_np(x: np.ndarray, y: np.ndarray, x_center: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    y = np.asarray(y)
    x_center = np.asarray(x_center)
    assert x.ndim == 2, x.ndim
    assert y.ndim in (1, 2), y.ndim
    assert x_center.ndim == 1, x_center.ndim
    assert x_center.shape[0] == x.shape[1], (x_center.shape, x.shape)
    if y.ndim == 2:
        assert y.shape[1] == 1, y.shape
        y = y[:, 0]
    assert len(x) == len(y), (len(x), len(y))

    # y - y.mean = beta * (x-x_center)**2 + eps
    xx = (x - x_center) ** 2
    y_centered = y - y.mean()

    cov_xy = (y_centered[:, None] * xx).mean(axis=0)
    var_xx = (xx**2).mean(axis=0)

    beta = np.zeros(x.shape[1])
    nonzero = var_xx > 1e-12
    beta[nonzero] = cov_xy[nonzero] / var_xx[nonzero]

    w = np.maximum(1e-6, np.abs(beta))
    w = w / w.sum()
    return w
