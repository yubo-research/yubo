from __future__ import annotations

import numpy as np
from torch import nn

from embedding.behavioral_embedder import BehavioralEmbedder

from .gaussian_perturbator import GaussianPerturbator
from .step_size_adapter import StepSizeAdapter


class _SimplePosterior:
    def __init__(self, *, mu: np.ndarray, se: np.ndarray):
        self.mu = mu
        self.se = se


class _SimpleENN:
    def __init__(self, *, x: np.ndarray, y: np.ndarray, k: int):
        self._x = np.asarray(x, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)
        self._k = int(max(1, min(k, x.shape[0])))

    def posterior(self, x_cand: np.ndarray) -> _SimplePosterior:
        x_cand = np.asarray(x_cand, dtype=np.float64)
        diff = x_cand[:, None, :] - self._x[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        idx = np.argpartition(d2, self._k - 1, axis=1)[:, : self._k]
        neigh = self._y[idx]
        mu = neigh.mean(axis=1)
        se = neigh.std(axis=1) / np.sqrt(self._k)
        return _SimplePosterior(mu=mu, se=se)


class UHDSimpleBE:
    """(1+1)-ES with ENN seed selection via behavioral embeddings."""

    def __init__(
        self,
        perturbator: GaussianPerturbator,
        sigma_0: float,
        dim: int,
        module: nn.Module,
        embedder: BehavioralEmbedder,
        *,
        num_candidates: int = 10,
        warmup: int = 20,
        fit_interval: int = 10,
        enn_k: int = 25,
    ):
        self._perturbator = perturbator
        self._adapter = StepSizeAdapter(sigma_0=sigma_0, dim=dim)
        self._module = module
        self._embedder = embedder
        self._num_candidates = num_candidates
        self._warmup = warmup
        self._fit_interval = fit_interval
        self._enn_k = enn_k

        self._next_seed = 0
        self._eval_seed = 0
        self._y_best: float | None = None
        self._mu_prev = 0.0
        self._se_prev = 0.0

        self._zs: list[np.ndarray] = []
        self._ys: list[float] = []
        self._enn_model: object | None = None
        self._enn_params: object | None = None
        self._posterior_flags: object | None = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self._num_new_since_fit = 0

    @property
    def eval_seed(self) -> int:
        return self._eval_seed

    @property
    def sigma(self) -> float:
        return self._adapter.sigma

    @property
    def y_best(self) -> float | None:
        return self._y_best

    @property
    def mu_avg(self) -> float:
        return self._mu_prev

    @property
    def se_avg(self) -> float:
        return self._se_prev

    def ask(self) -> None:
        if self._enn_params is not None and len(self._zs) >= self._warmup:
            self._eval_seed, self._z_current = self._select_seed()
            self._next_seed += self._num_candidates
        else:
            self._eval_seed = self._next_seed
            self._next_seed += 1
            self._perturbator.perturb(self._eval_seed, self._adapter.sigma)
            self._z_current = self._embed()

    def tell(self, mu: float, se: float) -> None:
        self._mu_prev = mu
        self._se_prev = se

        self._zs.append(self._z_current)
        self._ys.append(mu)
        self._num_new_since_fit += 1

        if self._y_best is None or mu > self._y_best:
            self._y_best = mu
            self._adapter.update(accepted=True)
            self._perturbator.accept()
        else:
            self._adapter.update(accepted=False)
            self._perturbator.unperturb()

        self._maybe_fit()

    def _embed(self) -> np.ndarray:
        was_training = self._module.training
        self._module.eval()
        z = self._embedder.embed(self._module)
        if was_training:
            self._module.train()
        return z.cpu().numpy().astype(np.float64)

    def _select_seed(self) -> tuple[int, np.ndarray]:
        base = self._next_seed
        was_training = self._module.training
        self._module.eval()

        zs = []
        for i in range(self._num_candidates):
            self._perturbator.perturb(base + i, self._adapter.sigma)
            zs.append(self._embedder.embed(self._module).cpu().numpy().astype(np.float64))
            self._perturbator.unperturb()

        if was_training:
            self._module.train()

        x_cand = np.array(zs, dtype=np.float64)
        mu_std, se_std = self._predict(x_cand)
        ucb = (self._y_mean + self._y_std * mu_std) + abs(self._y_std) * se_std
        best = int(np.argmax(ucb))

        self._perturbator.perturb(base + best, self._adapter.sigma)
        return base + best, zs[best]

    def _predict(self, x_cand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._posterior_flags is not None:
            post = self._enn_model.posterior(x_cand, params=self._enn_params, flags=self._posterior_flags)
        else:
            post = self._enn_model.posterior(x_cand)
        return np.asarray(post.mu).reshape(-1), np.asarray(post.se).reshape(-1)

    def _maybe_fit(self) -> None:
        if len(self._zs) < self._warmup:
            return
        if self._enn_params is not None and self._num_new_since_fit < self._fit_interval:
            return

        x = np.array(self._zs, dtype=np.float64)
        y = np.array(self._ys, dtype=np.float64)
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) if float(y.std()) > 0 else 1.0
        y_normed = (y - self._y_mean) / self._y_std

        try:
            from enn.enn.enn_class import EpistemicNearestNeighbors
            from enn.enn.enn_fit import enn_fit
            from enn.enn.enn_params import PosteriorFlags
            from enn.turbo.config.enn_index_driver import ENNIndexDriver

            self._enn_model = EpistemicNearestNeighbors(
                x,
                y_normed[:, None],
                None,
                scale_x=False,
                index_driver=ENNIndexDriver.FLAT,
            )
            self._enn_params = enn_fit(
                self._enn_model,
                k=int(self._enn_k),
                num_fit_candidates=200,
                num_fit_samples=200,
                rng=np.random.default_rng(0),
            )
            self._posterior_flags = PosteriorFlags(observation_noise=False)
        except Exception:
            self._enn_model = _SimpleENN(x=x, y=y_normed, k=int(self._enn_k))
            self._enn_params = True
            self._posterior_flags = None
        self._num_new_since_fit = 0
