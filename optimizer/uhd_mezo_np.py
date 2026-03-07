from __future__ import annotations

import math

import numpy as np

from embedding.behavioral_embedder import BehavioralEmbedder

from .uhd_simple_be import _fit_enn, _predict_enn


class UHDMeZONp:
    def __init__(
        self,
        policy,
        *,
        sigma: float = 0.001,
        lr: float = 0.001,
        beta: float = 0.9,
        param_clip: tuple[float, float] | None = None,
    ):
        self._policy = policy
        self._x = np.asarray(policy.get_params(), dtype=np.float64).copy()
        self._sigma = float(sigma)
        self._lr = float(lr)
        self._beta = float(beta)
        self._param_clip = param_clip

        self._seed = 0
        self._positive_phase = True
        self._step_seed = 0
        self._mu_plus = 0.0
        self._grad_sq_ema = 0.0
        self._last_step_scale = 0.0

        self._y_best: float | None = None
        self._mu_prev = 0.0
        self._se_prev = 0.0

    @property
    def eval_seed(self) -> int:
        return self._seed

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def y_best(self) -> float | None:
        return self._y_best

    @property
    def mu_avg(self) -> float:
        return self._mu_prev

    @property
    def se_avg(self) -> float:
        return self._se_prev

    @property
    def positive_phase(self) -> bool:
        return self._positive_phase

    def set_next_seed(self, seed: int) -> None:
        if not self._positive_phase:
            raise RuntimeError("set_next_seed is only valid during positive phase")
        self._seed = int(seed)

    def _noise(self, seed: int) -> np.ndarray:
        return np.random.default_rng(seed).standard_normal(len(self._x))

    def _clip(self, x: np.ndarray) -> np.ndarray:
        if self._param_clip is not None:
            return np.clip(x, self._param_clip[0], self._param_clip[1])
        return x

    def ask(self) -> None:
        z = self._noise(self._seed if self._positive_phase else self._step_seed)
        if self._positive_phase:
            self._step_seed = self._seed
            x_eval = self._clip(self._x + self._sigma * z)
        else:
            x_eval = self._clip(self._x - self._sigma * z)
        self._policy.set_params(x_eval)

    def tell(self, mu: float, se: float) -> None:
        self._mu_prev = float(mu)
        self._se_prev = float(se)
        if self._y_best is None or mu > self._y_best:
            self._y_best = float(mu)

        if self._positive_phase:
            self._mu_plus = float(mu)
            self._policy.set_params(self._x)
            self._positive_phase = False
            return

        mu_minus = float(mu)
        projected_grad = (self._mu_plus - mu_minus) / (2.0 * self._sigma)
        self._grad_sq_ema = self._beta * self._grad_sq_ema + (1.0 - self._beta) * projected_grad**2
        rms = math.sqrt(self._grad_sq_ema) + 1e-8
        step_scale = self._lr * projected_grad / rms
        self._last_step_scale = float(step_scale)

        z = self._noise(self._step_seed)
        self._x = self._clip(self._x + step_scale * z)
        self._policy.set_params(self._x)
        self._seed += 1
        self._positive_phase = True


class UHDMeZOBENp:
    def __init__(
        self,
        policy,
        embedder: BehavioralEmbedder,
        *,
        sigma: float = 0.001,
        lr: float = 0.001,
        beta: float = 0.9,
        param_clip: tuple[float, float] | None = None,
        num_candidates: int = 10,
        warmup: int = 20,
        fit_interval: int = 10,
        enn_k: int = 25,
    ):
        self._mezo = UHDMeZONp(policy, sigma=sigma, lr=lr, beta=beta, param_clip=param_clip)
        self._policy = policy
        self._embedder = embedder
        self._num_candidates = int(num_candidates)
        self._warmup = int(warmup)
        self._fit_interval = int(fit_interval)
        self._enn_k = int(enn_k)

        self._selected = False
        self._z_plus: np.ndarray | None = None
        self._z_minus: np.ndarray | None = None

        self._zs: list[np.ndarray] = []
        self._ys: list[float] = []
        self._enn_model: object | None = None
        self._enn_params: object | None = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self._num_new_since_fit = 0

    @property
    def eval_seed(self) -> int:
        return self._mezo.eval_seed

    @property
    def sigma(self) -> float:
        return self._mezo.sigma

    @property
    def y_best(self) -> float | None:
        return self._mezo.y_best

    @property
    def mu_avg(self) -> float:
        return self._mezo.mu_avg

    @property
    def se_avg(self) -> float:
        return self._mezo.se_avg

    @property
    def positive_phase(self) -> bool:
        return self._mezo.positive_phase

    def ask(self) -> None:
        if self._mezo.positive_phase:
            if self._enn_params is not None and len(self._zs) >= self._warmup:
                best_seed, z_plus, z_minus = self._select_seed()
                self._mezo.set_next_seed(best_seed)
                self._z_plus = z_plus
                self._z_minus = z_minus
                self._selected = True
            else:
                self._selected = False
            self._mezo.ask()
            if not self._selected:
                self._z_plus = self._embedder.embed_policy(self._policy, self._policy.get_params()).astype(np.float64)
        else:
            self._mezo.ask()
            if not self._selected:
                self._z_minus = self._embedder.embed_policy(self._policy, self._policy.get_params()).astype(np.float64)

    def tell(self, mu: float, se: float) -> None:
        is_positive = self._mezo.positive_phase
        z = self._z_plus if is_positive else self._z_minus
        self._zs.append(np.asarray(z, dtype=np.float64))
        self._ys.append(float(mu))
        self._num_new_since_fit += 1
        self._mezo.tell(mu, se)
        if not is_positive:
            self._maybe_fit()

    def _select_seed(self) -> tuple[int, np.ndarray, np.ndarray]:
        base = self._mezo.eval_seed
        sigma = self._mezo.sigma
        x_center = np.asarray(self._policy.get_params(), dtype=np.float64)
        dim = len(x_center)

        def _noise(seed: int) -> np.ndarray:
            return np.random.default_rng(seed).standard_normal(dim)

        z_plus_list = []
        z_minus_list = []
        for i in range(self._num_candidates):
            seed = base + i
            x_plus = x_center + sigma * _noise(seed)
            x_minus = x_center - sigma * _noise(seed)
            z_plus_list.append(self._embedder.embed_policy(self._policy, x_plus).astype(np.float64))
            z_minus_list.append(self._embedder.embed_policy(self._policy, x_minus).astype(np.float64))

        z_plus_arr = np.array(z_plus_list, dtype=np.float64)
        z_minus_arr = np.array(z_minus_list, dtype=np.float64)
        mu_plus, se_plus = _predict_enn(self._enn_model, self._enn_params, z_plus_arr)
        mu_minus, se_minus = _predict_enn(self._enn_model, self._enn_params, z_minus_arr)

        two_sigma = 2.0 * sigma
        g = (mu_plus - mu_minus) / two_sigma
        seg = np.sqrt(se_plus**2 + se_minus**2) / two_sigma
        ucb = np.abs(g) + seg
        best = int(np.argmax(ucb))
        return base + best, z_plus_list[best], z_minus_list[best]

    def _maybe_fit(self) -> None:
        if len(self._zs) < self._warmup:
            return
        if self._enn_params is not None and self._num_new_since_fit < self._fit_interval:
            return
        (
            self._enn_model,
            self._enn_params,
            self._y_mean,
            self._y_std,
        ) = _fit_enn(self._zs, self._ys, self._enn_k)
        self._num_new_since_fit = 0
