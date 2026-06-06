from __future__ import annotations

import numpy as np

from embedding.behavioral_embedder import BehavioralEmbedder

from .step_size_adapter import StepSizeAdapter
from .uhd_simple_be import _make_be_enn, _predict_enn, _tell_be_enn


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
        num_fit_candidates: int = 1,
        num_fit_samples: int = 10,
        enn_index_driver: str = "flat",
        adapt_sigma: bool = True,
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
        self._adapt_sigma = adapt_sigma

        self._next_seed = 0
        self._eval_seed = 0
        self._y_best: float | None = None
        self._mu_prev = 0.0
        self._se_prev = 0.0
        self._x_candidate: np.ndarray | None = None

        self._zs: list[np.ndarray] = []
        self._ys: list[float] = []
        self._be_enn = _make_be_enn(
            enn_k=enn_k,
            num_fit_candidates=num_fit_candidates,
            num_fit_samples=num_fit_samples,
            index_driver=enn_index_driver,
        )
        self._enn_model: object | None = None
        self._enn_params: object | None = None
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
            self._next_seed = self._eval_seed + 1
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
        _tell_be_enn(self, self._z_current, mu)

        if self._y_best is None or mu > self._y_best:
            self._y_best = mu
            if self._adapt_sigma:
                self._adapter.update(accepted=True)
            self._x = self._x_candidate.copy()
        else:
            if self._adapt_sigma:
                self._adapter.update(accepted=False)
            self._policy.set_params(self._x)

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
        mu_pred, se_pred = _predict_enn(self._enn_model, self._enn_params, z_cand)
        ucb = mu_pred + se_pred
        best = int(np.argmax(ucb))
        return base + best, candidates[best], embeddings[best]
