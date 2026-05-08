import numpy as np


class _Pert:
    def accept(self):
        return None

    def unperturb(self):
        return None


class _Policy:
    def __init__(self):
        self._x = np.zeros(3)

    def get_params(self):
        return self._x

    def set_params(self, x):
        self._x = np.asarray(x)


class _Embed:
    def embed_policy(self, _policy, x):
        return np.asarray(x, dtype=np.float64)
