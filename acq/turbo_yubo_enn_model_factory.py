import numpy as np
import torch

try:
    from model.enn import EpistemicNearestNeighbors  # uses faiss
except Exception:  # pragma: no cover

    class EpistemicNearestNeighbors:  # minimal fallback KNN without faiss
        def __init__(self, k=3, small_world_M=None):
            self.k = k
            self._x = None
            self._y = None

        def add(self, x, y):
            if self._x is None:
                self._x = np.array(x, dtype=float)
                self._y = np.array(y, dtype=float)
            else:
                self._x = np.append(self._x, np.array(x, dtype=float), axis=0)
                self._y = np.append(self._y, np.array(y, dtype=float), axis=0)

        def posterior(self, X):
            if self._x is None or len(self._x) == 0:
                mu = np.zeros((len(X), 1), dtype=float)
                se = np.ones((len(X), 1), dtype=float)
                return type("N", (), {"mu": mu, "se": se})
            X = np.asarray(X, dtype=float)
            diff = X[:, None, :] - self._x[None, :, :]
            dist2 = np.sum(diff * diff, axis=-1)
            k = min(self.k, self._x.shape[0])
            idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
            d2k = np.take_along_axis(dist2, idx, axis=1)
            yk = self._y[idx]
            w = 1.0 / (1e-9 + d2k)
            wsum = np.sum(w, axis=1, keepdims=True)
            mu = np.sum(w[:, :, None] * yk, axis=1) / wsum
            vvar = 1.0 / wsum
            se = np.sqrt(np.maximum(1e-9, vvar))
            return type("N", (), {"mu": mu, "se": se})


def _to_numpy(t: torch.Tensor):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t


def _to_torch(a: np.ndarray, *, like: torch.Tensor):
    return torch.as_tensor(a, dtype=like.dtype, device=like.device)


def build_turbo_yubo_enn_model(*, train_x: torch.Tensor, train_y: torch.Tensor, k: int = 3, small_world_M: int | None = None):
    x_t = train_x
    y_t = train_y
    if y_t.dim() > 1:
        y_t = y_t.squeeze(-1)

    x_np = _to_numpy(x_t)
    y_np = _to_numpy(y_t)[..., None]

    enn = EpistemicNearestNeighbors(k=k, small_world_M=small_world_M)
    enn.add(x_np, y_np)

    class _ENNModel:
        def __init__(self, x_like: torch.Tensor, y_like: torch.Tensor):
            self.train_inputs = (x_like.detach(),)
            self.train_targets = y_like.detach()
            self.covar_module = None

        def posterior(self, X: torch.Tensor):
            X_np = _to_numpy(X)
            mvn = enn.posterior(X_np)

            class _P:
                def __init__(self, X_like: torch.Tensor, mu: np.ndarray, se: np.ndarray):
                    self._X_like = X_like
                    self._mu = mu
                    self._se = se

                def sample(self, sample_shape: torch.Size):
                    n = sample_shape[0] if isinstance(sample_shape, torch.Size) else int(sample_shape)
                    eps = np.random.normal(size=(self._mu.shape[0], n))
                    samples = self._mu[:, 0:1] + self._se[:, 0:1] * eps
                    samples_t = _to_torch(samples[:, :, None], like=self._X_like)
                    return samples_t.permute(1, 0, 2).contiguous()

            return _P(X, mvn.mu, mvn.se)

    return _ENNModel(x_t, y_t)
