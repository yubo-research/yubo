"""Incremental ENN regressor for behavioral-embedder seed selection."""

from __future__ import annotations

import numpy as np
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_fit import enn_fit
from enn.enn.enn_fitter import ENNStatefulFitter
from enn.enn.enn_params import PosteriorFlags
from enn.turbo.config.enn_index_driver import ENNIndexDriver

from .uhd_enn_fit_helpers import fit_enn_params

_BATCH_FIT_CANDIDATES = 200
_BATCH_FIT_SAMPLES = 200


def parse_be_enn_index_driver(name: str) -> ENNIndexDriver:
    key = str(name).strip().lower()
    if key == "hnsw":
        return ENNIndexDriver.HNSW
    if key in {"hnsw_disk", "hnsw-disk"}:
        return ENNIndexDriver.HNSW_DISK
    if key != "flat":
        raise ValueError(f"Unknown be_enn_index_driver: {name!r}")
    return ENNIndexDriver.FLAT


class IncrementalBEEnn:
    """Raw-y ENN for BE: model.add + batch enn_fit ask synced to model rows."""

    def __init__(
        self,
        *,
        k: int,
        num_fit_candidates: int = 1,
        num_fit_samples: int = 10,
        fit_interval: int = 10,
        batch_fit_candidates: int = _BATCH_FIT_CANDIDATES,
        batch_fit_samples: int = _BATCH_FIT_SAMPLES,
        index_driver: ENNIndexDriver | str = ENNIndexDriver.FLAT,
        rng: np.random.Generator | None = None,
    ):
        if isinstance(index_driver, str):
            index_driver = parse_be_enn_index_driver(index_driver)
        self._k = int(k)
        self._num_fit_candidates = int(num_fit_candidates)
        self._num_fit_samples = int(num_fit_samples)
        self._fit_interval = max(1, int(fit_interval))
        self._batch_fit_candidates = int(batch_fit_candidates)
        self._batch_fit_samples = int(batch_fit_samples)
        self._index_driver = index_driver
        self._rng = rng if rng is not None else np.random.default_rng(0)
        self._model: EpistemicNearestNeighbors | None = None
        self._fitter: ENNStatefulFitter | None = None
        self._params = None
        self._embed_dim: int | None = None
        self._num_since_heavy_fit = 0

    @classmethod
    def create_empty(
        cls,
        embed_dim: int,
        *,
        k: int,
        num_fit_candidates: int = 1,
        num_fit_samples: int = 10,
        fit_interval: int = 10,
        index_driver: ENNIndexDriver | str = ENNIndexDriver.FLAT,
        rng: np.random.Generator | None = None,
    ) -> IncrementalBEEnn:
        if isinstance(index_driver, str):
            index_driver = parse_be_enn_index_driver(index_driver)
        reg = cls(
            k=k,
            num_fit_candidates=num_fit_candidates,
            num_fit_samples=num_fit_samples,
            fit_interval=fit_interval,
            index_driver=index_driver,
            rng=rng,
        )
        reg._init_empty_model(int(embed_dim))
        return reg

    def _init_empty_model(self, embed_dim: int) -> None:
        self._embed_dim = int(embed_dim)
        self._model = EpistemicNearestNeighbors(
            np.empty((0, self._embed_dim), dtype=np.float64),
            np.empty((0, 1), dtype=np.float64),
            None,
            scale_x=False,
            index_driver=self._index_driver,
        )
        self._fitter = ENNStatefulFitter(
            k=self._k,
            rng=self._rng,
            infer_aleatoric_variance_scale=True,
        )

    @property
    def model(self) -> EpistemicNearestNeighbors | None:
        return self._model

    @property
    def params(self):
        return self._params

    @property
    def obs_count(self) -> int:
        if self._model is None:
            return 0
        return len(self._model)

    def _effective_k(self) -> int:
        n = self.obs_count
        if n <= 1:
            return 1
        return min(self._k, n - 1)

    def _ensure_model(self, embed_dim: int) -> None:
        if self._model is None:
            self._init_empty_model(embed_dim)

    def add_obs(self, z: np.ndarray, y: float) -> None:
        row_x = np.asarray(z, dtype=np.float64).reshape(1, -1)
        row_y = np.asarray([[float(y)]], dtype=np.float64)
        self._ensure_model(int(row_x.shape[1]))
        assert self._model is not None

        self._model.add(row_x, row_y)
        self._num_since_heavy_fit += 1

        heavy = self._params is None or self._num_since_heavy_fit >= self._fit_interval
        if heavy:
            num_fit_candidates = self._batch_fit_candidates
            num_fit_samples = self._batch_fit_samples
            self._num_since_heavy_fit = 0
        else:
            num_fit_candidates = self._num_fit_candidates
            num_fit_samples = self._num_fit_samples

        self._params = enn_fit(
            self._model,
            k=self._effective_k(),
            num_fit_candidates=num_fit_candidates,
            num_fit_samples=num_fit_samples,
            rng=self._rng,
            params_warm_start=self._params,
            incremental=None,
        )
        self._resync_fitter_after_fit()

    def _resync_fitter_after_fit(self) -> None:
        assert self._model is not None
        self._fitter = ENNStatefulFitter(
            k=self._k,
            rng=self._rng,
            infer_aleatoric_variance_scale=True,
        )
        if len(self._model) == 0:
            return
        idx = list(range(len(self._model)))
        x_all, y_all, yvar = self._model.train_rows_at(idx)
        self._fitter.tell(x_all, y_all, yvar)

    def predict(self, x_cand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._model is None or self._params is None:
            raise RuntimeError("IncrementalBEEnn has no observations yet")
        post = self._model.posterior(
            np.asarray(x_cand, dtype=np.float64),
            params=self._params,
            flags=PosteriorFlags(observation_noise=False),
        )
        return np.asarray(post.mu).reshape(-1), np.asarray(post.se).reshape(-1)


def fit_enn_batch_raw_reference(zs, ys, enn_k, *, num_fit_candidates=1, num_fit_samples=10):
    """One-shot batch fit on raw y (reference for incremental metamorphic tests)."""
    x = np.array(zs, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)
    model = EpistemicNearestNeighbors(
        x,
        y[:, None],
        None,
        scale_x=False,
        index_driver=ENNIndexDriver.FLAT,
    )
    params = fit_enn_params(
        model,
        x,
        y,
        k=int(enn_k),
        num_fit_candidates=num_fit_candidates,
        num_fit_samples=num_fit_samples,
        rng=np.random.default_rng(0),
    )
    return model, params


def fit_enn_batch_reference(zs, ys, enn_k, *, num_fit_candidates=200, num_fit_samples=200):
    """Batch ENN fit with y normalization (legacy reference)."""
    x = np.array(zs, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)
    y_mean = float(y.mean())
    y_std = float(y.std()) if float(y.std()) > 0 else 1.0
    y_normed = (y - y_mean) / y_std

    model = EpistemicNearestNeighbors(
        x,
        y_normed[:, None],
        None,
        scale_x=False,
        index_driver=ENNIndexDriver.FLAT,
    )
    params = fit_enn_params(
        model,
        x,
        y_normed,
        k=int(enn_k),
        num_fit_candidates=num_fit_candidates,
        num_fit_samples=num_fit_samples,
        rng=np.random.default_rng(0),
    )
    return model, params, y_mean, y_std


def ucb_from_batch_posterior(model, params, x_cand, y_mean, y_std):
    post = model.posterior(
        np.asarray(x_cand, dtype=np.float64),
        params=params,
        flags=PosteriorFlags(observation_noise=False),
    )
    mu_std = np.asarray(post.mu).reshape(-1)
    se_std = np.asarray(post.se).reshape(-1)
    return (y_mean + y_std * mu_std) + abs(y_std) * se_std


def ucb_from_incremental(reg: IncrementalBEEnn, x_cand: np.ndarray) -> np.ndarray:
    mu, se = reg.predict(x_cand)
    return mu + se


def be_enn_selection_ready(*, obs_count: int, warmup: int, enn_k: int, has_params: bool) -> bool:
    del enn_k  # k is capped inside IncrementalBEEnn from obs_count
    return has_params and obs_count >= int(warmup)
