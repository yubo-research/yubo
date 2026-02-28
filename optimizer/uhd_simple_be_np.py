from __future__ import annotations

import numpy as np
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_fit import enn_fit
from enn.enn.enn_params import PosteriorFlags
from enn.turbo.config.enn_index_driver import ENNIndexDriver

from embedding.behavioral_embedder import BehavioralEmbedder

from .step_size_adapter import StepSizeAdapter


class UHDSimpleBENp:
    """(1+1)-ES with ENN seed selection via behavioral embeddings for numpy policies."""

    def __init__(
        self,
        policy,
        embedder: BehavioralEmbedder,
        *,
        sigma_0: float,
        param_clip: tuple[float, float] | None = None,
        num_candidates: int = 10,
        warmup: int = 20,
        fit_interval: int = 10,
        enn_k: int = 25,
    ):
        self._policy = policy
        self._embedder = embedder
        self._x = np.asarray(policy.get_params(), dtype=np.float64).copy()
        dim = len(self._x)
        self._adapter = StepSizeAdapter(sigma_0=sigma_0, dim=dim)
        self._param_clip = param_clip
        self._num_candidates = num_candidates
        self._warmup = warmup
        self._fit_interval = fit_interval
        self._enn_k = enn_k

        self._next_seed = 0
        self._eval_seed = 0
        self._y_best: float | None = None
        self._mu_prev = 0.0
        self._se_prev = 0.0
        self._x_candidate: np.ndarray | None = None

        self._zs: list[np.ndarray] = []
        self._ys: list[float] = []
        self._enn_model: object | None = None
        self._enn_params: object | None = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self._num_new_since_fit = 0
        self._z_current: np.ndarray | None = None

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

    def _noise(self, seed: int) -> np.ndarray:
        return np.random.default_rng(seed).standard_normal(len(self._x))

    def _clip(self, x: np.ndarray) -> np.ndarray:
        if self._param_clip is not None:
            return np.clip(x, self._param_clip[0], self._param_clip[1])
        return x

    def ask(self) -> None:
        if self._enn_params is not None and len(self._zs) >= self._warmup:
            self._eval_seed, self._x_candidate, self._z_current = self._select_seed()
            self._next_seed += self._num_candidates
        else:
            self._eval_seed = self._next_seed
            self._next_seed += 1
            self._x_candidate = self._clip(self._x + self._adapter.sigma * self._noise(self._eval_seed))
            self._z_current = self._embedder.embed_policy(self._policy, self._x_candidate)
        self._policy.set_params(self._x_candidate)

    def tell(self, mu: float, se: float) -> None:
        self._mu_prev = mu
        self._se_prev = se

        self._zs.append(self._z_current)
        self._ys.append(mu)
        self._num_new_since_fit += 1

        if self._y_best is None or mu > self._y_best:
            self._y_best = mu
            self._adapter.update(accepted=True)
            self._x = self._x_candidate.copy()
        else:
            self._adapter.update(accepted=False)
            self._policy.set_params(self._x)

        self._maybe_fit()

    def _select_seed(self) -> tuple[int, np.ndarray, np.ndarray]:
        base = self._next_seed
        candidates = []
        embeddings = []
        for i in range(self._num_candidates):
            x_c = self._clip(self._x + self._adapter.sigma * self._noise(base + i))
            z_c = self._embedder.embed_policy(self._policy, x_c)
            candidates.append(x_c)
            embeddings.append(z_c)

        z_cand = np.array(embeddings, dtype=np.float64)
        mu_std, se_std = self._predict(z_cand)
        ucb = (self._y_mean + self._y_std * mu_std) + abs(self._y_std) * se_std
        best = int(np.argmax(ucb))
        return base + best, candidates[best], embeddings[best]

    def _predict(self, x_cand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        post = self._enn_model.posterior(x_cand, params=self._enn_params, flags=PosteriorFlags(observation_noise=False))
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
        self._num_new_since_fit = 0
