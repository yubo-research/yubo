import numpy as np

from model.enn import ENNNormal, EpistemicNearestNeighbors


class ENNMulti:
    def __init__(self, ks: list[int]):
        self.ks = ks
        self._enn = EpistemicNearestNeighbors(k=max(ks), small_world_M=None)
        self._beta = None

    def add(self, x: np.ndarray, y: np.ndarray):
        self._enn.add(x, y)

        if len(self._enn) > 0:
            mp = self._enn.multi_posterior(x, ks=self.ks)

            num_metrics = y.shape[1]
            beta = np.zeros((len(self.ks), num_metrics))

            for m in range(num_metrics):
                X = mp.mu[:, :, m]
                Y = y[:, m]
                try:
                    beta[:, m] = np.linalg.lstsq(X, Y, rcond=None)[0]
                except np.linalg.LinAlgError:
                    print("Warning: Couldn't fit")
                    beta[:, m] = np.zeros(len(self.ks))
                    if len(self.ks) > 0:
                        beta[-1, m] = 1.0

            self._beta = beta

    def posterior(self, x: np.ndarray) -> ENNNormal:
        if self._beta is None:
            batch_size = x.shape[0]
            num_metrics = self._enn._num_metrics if hasattr(self._enn, "_num_metrics") and self._enn._num_metrics is not None else 1
            mu = np.zeros((batch_size, num_metrics))
            se = np.ones((batch_size, num_metrics))
            return ENNNormal(mu, se)

        mp = self._enn.multi_posterior(x, ks=self.ks)

        batch_size = x.shape[0]
        num_metrics = mp.mu.shape[2]

        mu = np.zeros((batch_size, num_metrics))
        se_sq = np.zeros((batch_size, num_metrics))

        for m in range(num_metrics):
            mu_raw = mp.mu[:, :, m] @ self._beta[:, m]
            se_raw = mp.se[:, :, m]

            mask_finite = np.isfinite(se_raw) & (se_raw > 0)
            beta_finite = np.where(mask_finite.any(axis=0), self._beta[:, m], 0)

            if mask_finite.sum() > 0:
                mu[:, m] = np.where(np.isfinite(mu_raw), mu_raw, 0)
                se_sq_finite = np.where(mask_finite, se_raw**2, 0)
                se_sq[:, m] = se_sq_finite @ (beta_finite**2)
                se_sq[:, m] = np.maximum(se_sq[:, m], 1e-10)
            else:
                mu[:, m] = 0
                se_sq[:, m] = 1.0

        return ENNNormal(mu, np.sqrt(se_sq))
