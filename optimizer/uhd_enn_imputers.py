from __future__ import annotations

from typing import Any

import numpy as np

from optimizer.uhd_enn_regression import fit_enn, predict_enn, sample_objective_noise


_VALID_TARGETS = {"mu_minus", "delta", "mu_plus"}


class _JAXImputerBase:
    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._target = str(cfg.target)
        if self._target not in _VALID_TARGETS:
            raise ValueError("enn_target must be one of 'mu_minus', 'delta', or 'mu_plus'.")

        self._zs: list[np.ndarray] = []
        self._ys: list[float] = []
        self._model = None
        self._params = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self._num_new_since_fit = 0
        self._num_real_evals = 0
        self._num_imputed = 0
        self._abs_err_ema: float | None = None
        self._num_calib = 0

    @property
    def num_real_evals(self) -> int:
        return self._num_real_evals

    @property
    def num_imputed(self) -> int:
        return self._num_imputed

    @property
    def abs_err_ema(self) -> float | None:
        return self._abs_err_ema

    def _add_observation(self, z: np.ndarray, y: float) -> None:
        self._zs.append(np.asarray(z, dtype=np.float64))
        self._ys.append(float(y))
        self._num_new_since_fit += 1

    def _maybe_fit(self) -> None:
        if len(self._zs) < max(2, int(self._cfg.warmup_real_obs)):
            return
        if self._params is not None and self._num_new_since_fit < int(self._cfg.fit_interval):
            return
        self._model, self._params, self._y_mean, self._y_std = fit_enn(self._zs, self._ys, self._cfg.k)
        self._num_new_since_fit = 0

    def _fit_ready(self, phase_attr: str) -> bool:
        setattr(self, phase_attr, getattr(self, phase_attr) + 1)
        if len(self._zs) < int(self._cfg.warmup_real_obs):
            return False
        phase_count = getattr(self, phase_attr)
        if int(self._cfg.refresh_interval) > 0 and phase_count % int(self._cfg.refresh_interval) == 0:
            return False
        self._maybe_fit()
        if self._model is None or self._params is None:
            return False
        if self._num_calib < int(self._cfg.min_calib_points):
            return False
        if self._abs_err_ema is None:
            return False
        return float(self._abs_err_ema) <= float(self._cfg.max_abs_err_ema)

    def _predict_feature_y(self, z: np.ndarray, *, not_fit_msg: str) -> tuple[float, float]:
        if self._model is None or self._params is None:
            raise RuntimeError(not_fit_msg)
        mu_std, se_std = predict_enn(self._model, self._params, np.asarray([z], dtype=np.float64))
        y_hat = float(self._y_mean + self._y_std * float(mu_std[0]))
        y_se = float(abs(self._y_std) * float(se_std[0]))
        return y_hat, y_se

    def _update_abs_err_ema(self, err: float) -> None:
        if self._abs_err_ema is None:
            self._abs_err_ema = float(err)
        else:
            beta = float(self._cfg.err_ema_beta)
            self._abs_err_ema = beta * float(self._abs_err_ema) + (1.0 - beta) * float(err)
        self._num_calib += 1


class JAXMinusImputer(_JAXImputerBase):
    """ENN minus imputer over EggRoll/JAX behavior embeddings."""

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._num_negative_phases = 0
        self._num_choose_calls = 0
        self._num_selected = 0
        self._last_mu_plus: float | None = None

    @property
    def num_selected(self) -> int:
        return self._num_selected

    def tell_plus(self, *, z_plus: np.ndarray, z_minus: np.ndarray, mu_plus: float) -> None:
        self._last_mu_plus = float(mu_plus)
        z_plus = np.asarray(z_plus, dtype=np.float64)
        z_minus = np.asarray(z_minus, dtype=np.float64)
        if self._target == "mu_plus":
            self._add_observation(z_plus - z_minus, float(mu_plus))
        elif self._target != "delta":
            self._add_observation(z_plus, float(mu_plus))
        self._num_real_evals += 1

    def tell_real_minus(self, *, z_minus: np.ndarray, z_delta: np.ndarray, mu_minus: float) -> None:
        if self._target == "mu_plus":
            self._num_real_evals += 1
            return
        if self._target == "delta":
            if self._last_mu_plus is None:
                raise RuntimeError("enn_target='delta' requires a preceding real mu_plus in the same pair.")
            self._add_observation(z_delta, float(self._last_mu_plus) - float(mu_minus))
        else:
            self._add_observation(z_minus, float(mu_minus))
        self._num_real_evals += 1

    def _predict_y(self, *, z_minus: np.ndarray, z_delta: np.ndarray) -> tuple[float, float]:
        z = z_delta if self._target in {"delta", "mu_plus"} else z_minus
        return self._predict_feature_y(z, not_fit_msg="ENN imputer is not fitted.")

    def predict_minus(self, *, z_minus: np.ndarray, z_delta: np.ndarray) -> tuple[float, float]:
        y_hat, y_se = self._predict_y(z_minus=z_minus, z_delta=z_delta)
        if self._target == "delta":
            if self._last_mu_plus is None:
                raise RuntimeError("enn_target='delta' requires a preceding real mu_plus in the same pair.")
            return float(self._last_mu_plus) - float(y_hat), float(y_se)
        return float(y_hat), float(y_se)

    def calibrate_minus(self, *, z_minus: np.ndarray, z_delta: np.ndarray, mu_minus_real: float) -> None:
        if self._target == "mu_plus" or self._model is None or self._params is None:
            return
        try:
            mu_hat, _se_hat = self.predict_minus(z_minus=z_minus, z_delta=z_delta)
        except Exception:
            return
        self._update_abs_err_ema(abs(float(mu_minus_real) - float(mu_hat)))

    def _should_impute_negative(self) -> bool:
        if self._target == "mu_plus":
            return False
        return self._fit_ready("_num_negative_phases")

    def try_impute_minus(self, *, z_minus: np.ndarray, z_delta: np.ndarray) -> tuple[bool, float, float]:
        if not self._should_impute_negative():
            return False, float("nan"), float("nan")
        mu, se = self.predict_minus(z_minus=z_minus, z_delta=z_delta)
        if float(se) > float(self._cfg.se_threshold):
            return False, float("nan"), float("nan")
        self._num_imputed += 1
        return True, float(mu), float(se)

    def choose_seed_ucb(
        self,
        *,
        objective,
        cfg,
        x: np.ndarray,
        base_seed: int,
    ) -> tuple[int, tuple[np.ndarray, np.ndarray] | None]:
        self._num_choose_calls += 1
        if int(self._cfg.select_interval) > 1 and self._num_choose_calls % int(self._cfg.select_interval) != 0:
            return int(base_seed), None
        if self._target not in {"delta", "mu_plus"} or int(self._cfg.num_candidates) <= 1:
            return int(base_seed), None
        self._maybe_fit()
        if self._model is None or self._params is None:
            return int(base_seed), None

        seeds = [int(base_seed) + j for j in range(int(self._cfg.num_candidates))]
        noises = [sample_objective_noise(objective, cfg, s) for s in seeds]
        x_plus = np.stack([x + float(cfg.sigma) * n for n in noises])
        x_minus = np.stack([x - float(cfg.sigma) * n for n in noises])
        z_pair = objective.embed_many(np.concatenate([x_plus, x_minus], axis=0))
        z_plus = z_pair[: len(seeds)]
        z_minus = z_pair[len(seeds) :]
        z_delta = z_plus - z_minus
        mu_std, se_std = predict_enn(self._model, self._params, z_delta)
        mu = self._y_mean + self._y_std * mu_std
        se = abs(self._y_std) * se_std
        best = int(np.argmax(mu + se))
        if seeds[best] != int(base_seed):
            self._num_selected += 1
        return seeds[best], (z_plus[best], z_minus[best])


class JAXPointImputer(_JAXImputerBase):
    """ENN imputer for one-sided EggRoll/JAX point evaluations."""

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._num_candidate_phases = 0

    def _feature(self, *, z_eval: np.ndarray, z_base: np.ndarray) -> np.ndarray:
        if self._target == "delta":
            return np.asarray(z_eval, dtype=np.float64) - np.asarray(z_base, dtype=np.float64)
        return np.asarray(z_eval, dtype=np.float64)

    def _target_value(self, *, mu_eval: float, mu0: float, epsilon: float) -> float:
        if self._target == "delta":
            return (float(mu_eval) - float(mu0)) / float(epsilon)
        return float(mu_eval)

    def tell_base(self, *, z_base: np.ndarray, mu0: float) -> None:
        if self._target != "delta":
            self._add_observation(np.asarray(z_base, dtype=np.float64), float(mu0))
        self._num_real_evals += 1

    def tell_real_eval(
        self,
        *,
        z_eval: np.ndarray,
        z_base: np.ndarray,
        mu_eval: float,
        mu0: float,
        epsilon: float,
    ) -> None:
        z = self._feature(z_eval=z_eval, z_base=z_base)
        y = self._target_value(mu_eval=float(mu_eval), mu0=float(mu0), epsilon=float(epsilon))
        self._add_observation(z, y)
        self._num_real_evals += 1

    def _predict_y(self, *, z_eval: np.ndarray, z_base: np.ndarray) -> tuple[float, float]:
        z = self._feature(z_eval=z_eval, z_base=z_base)
        return self._predict_feature_y(z, not_fit_msg="ENN point imputer is not fitted.")

    def predict_mu(self, *, z_eval: np.ndarray, z_base: np.ndarray, mu0: float, epsilon: float) -> tuple[float, float]:
        y_hat, y_se = self._predict_y(z_eval=z_eval, z_base=z_base)
        if self._target == "delta":
            return float(mu0) + float(epsilon) * float(y_hat), abs(float(epsilon)) * float(y_se)
        return float(y_hat), float(y_se)

    def calibrate_eval(
        self,
        *,
        z_eval: np.ndarray,
        z_base: np.ndarray,
        mu_eval_real: float,
        mu0: float,
        epsilon: float,
    ) -> None:
        if self._model is None or self._params is None:
            return
        try:
            mu_hat, _se_hat = self.predict_mu(z_eval=z_eval, z_base=z_base, mu0=float(mu0), epsilon=float(epsilon))
        except Exception:
            return
        self._update_abs_err_ema(abs(float(mu_eval_real) - float(mu_hat)))

    def _should_impute_eval(self) -> bool:
        return self._fit_ready("_num_candidate_phases")

    def try_impute_eval(self, *, z_eval: np.ndarray, z_base: np.ndarray, mu0: float, epsilon: float) -> tuple[bool, float, float]:
        if not self._should_impute_eval():
            return False, float("nan"), float("nan")
        mu, se = self.predict_mu(z_eval=z_eval, z_base=z_base, mu0=float(mu0), epsilon=float(epsilon))
        if float(se) > float(self._cfg.se_threshold):
            return False, float("nan"), float("nan")
        self._num_imputed += 1
        return True, float(mu), float(se)


def format_enn_stats(imputer: Any, *, label: str = "imputed_minus") -> str:
    err = imputer.abs_err_ema
    err_s = "N/A" if err is None else f"{err:.4f}"
    extra = ""
    num_candidates = int(getattr(getattr(imputer, "_cfg", object()), "num_candidates", 1))
    if hasattr(imputer, "num_selected") and num_candidates > 1:
        extra = f" seedselect={imputer.num_selected}"
    return f"ENN: real_evals={imputer.num_real_evals} {label}={imputer.num_imputed} abs_err_ema={err_s}{extra}"
