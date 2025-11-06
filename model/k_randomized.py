import numpy as np
from scipy.stats import qmc


class KRandomized:
    def __init__(self, train_x: np.ndarray):
        assert isinstance(train_x, np.ndarray)
        assert train_x.ndim == 2
        n, d = train_x.shape
        assert n >= 1 and d >= 1
        self._train_x = np.asarray(train_x, dtype=np.float64)
        sobol = qmc.Sobol(d, scramble=True)
        u = sobol.random(1).reshape(d)
        l_min = 1e-2
        l_max = 1.0
        self._lengthscales = np.exp(np.log(l_min) + u * (np.log(l_max) - np.log(l_min)))

        Xs = self._train_x / self._lengthscales
        xs2 = (Xs**2).sum(axis=1)
        D2 = xs2[:, None] + xs2[None, :] - 2.0 * (Xs @ Xs.T)
        D2 = np.maximum(D2, 0.0)
        K = np.exp(-0.5 * D2)
        K = 0.5 * (K + K.T)
        np.fill_diagonal(K, 1.0)
        self._K = K

    def sub_k(self, idxs: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert isinstance(idxs, np.ndarray)
        assert idxs.ndim == 1
        assert isinstance(x, np.ndarray)
        assert x.ndim in (1, 2)

        idxs = idxs.astype(int, copy=False)
        Kxx = self._K[np.ix_(idxs, idxs)]

        Xq = x.astype(np.float64, copy=False)
        if Xq.ndim == 1:
            Xq = Xq[None, :]
        b, d = Xq.shape
        assert d == self._train_x.shape[1]

        Z = self._train_x[idxs]
        Q = Xq / self._lengthscales
        Zs = Z / self._lengthscales

        q2 = (Q**2).sum(axis=1)
        z2 = (Zs**2).sum(axis=1)
        D2 = q2[:, None] + z2[None, :] - 2.0 * (Q @ Zs.T)
        D2 = np.maximum(D2, 0.0)
        Kx = np.exp(-0.5 * D2)

        if x.ndim == 1:
            Kx = Kx[0]

        return Kxx, Kx
