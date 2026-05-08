from __future__ import annotations

import numpy as np


def fit_enn(zs: list[np.ndarray], ys: list[float], enn_k: int):
    from optimizer.uhd_simple_be import _fit_enn

    return _fit_enn(zs, ys, int(enn_k))


def predict_enn(model, params, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from optimizer.uhd_simple_be import _predict_enn

    return _predict_enn(model, params, x)


def fit_if_due(state: dict, cfg, *, require_pair: bool = False) -> None:
    if len(state["zs"]) < int(cfg.be.warmup):
        return
    if require_pair and state.get("phase_since_fit", 0) <= 0:
        return
    if state["params"] is not None and state["new_since_fit"] < int(cfg.be.fit_interval):
        return
    model, params, y_mean, y_std = fit_enn(state["zs"], state["ys"], cfg.be.enn_k)
    state["model"] = model
    state["params"] = params
    state["y_mean"] = y_mean
    state["y_std"] = y_std
    state["new_since_fit"] = 0
    state["phase_since_fit"] = 0


def new_be_state() -> dict:
    return {
        "zs": [],
        "ys": [],
        "model": None,
        "params": None,
        "y_mean": 0.0,
        "y_std": 1.0,
        "new_since_fit": 0,
        "phase_since_fit": 0,
    }


def predict_real_ucb(state: dict, embeddings: np.ndarray) -> np.ndarray:
    mu_std, se_std = predict_enn(state["model"], state["params"], embeddings)
    return (state["y_mean"] + state["y_std"] * mu_std) + abs(state["y_std"]) * se_std


def _sample_objective_noise(objective, cfg, seed: int) -> np.ndarray:
    return objective.sample_noise(
        seed=int(seed),
        num_dim_target=cfg.num_dim_target,
        num_module_target=cfg.num_module_target,
    )


class JAXMinusImputer:
    """ENN minus imputer over EggRoll/JAX behavior embeddings."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._target = str(cfg.target)
        if self._target not in {"mu_minus", "delta", "mu_plus"}:
            raise ValueError("enn_target must be one of 'mu_minus', 'delta', or 'mu_plus'.")

        self._zs: list[np.ndarray] = []
        self._ys: list[float] = []
        self._model = None
        self._params = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self._num_new_since_fit = 0
        self._num_negative_phases = 0
        self._num_choose_calls = 0
        self._num_real_evals = 0
        self._num_imputed = 0
        self._num_selected = 0
        self._abs_err_ema: float | None = None
        self._num_calib = 0
        self._last_mu_plus: float | None = None

    @property
    def num_real_evals(self) -> int:
        return self._num_real_evals

    @property
    def num_imputed(self) -> int:
        return self._num_imputed

    @property
    def num_selected(self) -> int:
        return self._num_selected

    @property
    def abs_err_ema(self) -> float | None:
        return self._abs_err_ema

    def _add_observation(self, z: np.ndarray, y: float) -> None:
        self._zs.append(np.asarray(z, dtype=np.float64))
        self._ys.append(float(y))
        self._num_new_since_fit += 1

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

    def _maybe_fit(self) -> None:
        if len(self._zs) < max(2, int(self._cfg.warmup_real_obs)):
            return
        if self._params is not None and self._num_new_since_fit < int(self._cfg.fit_interval):
            return
        self._model, self._params, self._y_mean, self._y_std = fit_enn(self._zs, self._ys, self._cfg.k)
        self._num_new_since_fit = 0

    def _predict_y(self, *, z_minus: np.ndarray, z_delta: np.ndarray) -> tuple[float, float]:
        if self._model is None or self._params is None:
            raise RuntimeError("ENN imputer is not fitted.")
        z = z_delta if self._target in {"delta", "mu_plus"} else z_minus
        mu_std, se_std = predict_enn(self._model, self._params, np.asarray([z], dtype=np.float64))
        y_hat = float(self._y_mean + self._y_std * float(mu_std[0]))
        y_se = float(abs(self._y_std) * float(se_std[0]))
        return y_hat, y_se

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
        err = abs(float(mu_minus_real) - float(mu_hat))
        if self._abs_err_ema is None:
            self._abs_err_ema = float(err)
        else:
            beta = float(self._cfg.err_ema_beta)
            self._abs_err_ema = beta * float(self._abs_err_ema) + (1.0 - beta) * float(err)
        self._num_calib += 1

    def _should_impute_negative(self) -> bool:
        if self._target == "mu_plus":
            return False
        self._num_negative_phases += 1
        if len(self._zs) < int(self._cfg.warmup_real_obs):
            return False
        if int(self._cfg.refresh_interval) > 0 and self._num_negative_phases % int(self._cfg.refresh_interval) == 0:
            return False
        self._maybe_fit()
        if self._model is None or self._params is None:
            return False
        if self._num_calib < int(self._cfg.min_calib_points):
            return False
        if self._abs_err_ema is None:
            return False
        return float(self._abs_err_ema) <= float(self._cfg.max_abs_err_ema)

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
        noises = [_sample_objective_noise(objective, cfg, s) for s in seeds]
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


class JAXPointImputer:
    """ENN imputer for one-sided EggRoll/JAX point evaluations."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._target = str(cfg.target)
        if self._target not in {"mu_minus", "delta", "mu_plus"}:
            raise ValueError("enn_target must be one of 'mu_minus', 'delta', or 'mu_plus'.")

        self._zs: list[np.ndarray] = []
        self._ys: list[float] = []
        self._model = None
        self._params = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self._num_new_since_fit = 0
        self._num_candidate_phases = 0
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

    def tell_real_eval(self, *, z_eval: np.ndarray, z_base: np.ndarray, mu_eval: float, mu0: float, epsilon: float) -> None:
        z = self._feature(z_eval=z_eval, z_base=z_base)
        y = self._target_value(mu_eval=float(mu_eval), mu0=float(mu0), epsilon=float(epsilon))
        self._add_observation(z, y)
        self._num_real_evals += 1

    def _maybe_fit(self) -> None:
        if len(self._zs) < max(2, int(self._cfg.warmup_real_obs)):
            return
        if self._params is not None and self._num_new_since_fit < int(self._cfg.fit_interval):
            return
        self._model, self._params, self._y_mean, self._y_std = fit_enn(self._zs, self._ys, self._cfg.k)
        self._num_new_since_fit = 0

    def _predict_y(self, *, z_eval: np.ndarray, z_base: np.ndarray) -> tuple[float, float]:
        if self._model is None or self._params is None:
            raise RuntimeError("ENN point imputer is not fitted.")
        z = self._feature(z_eval=z_eval, z_base=z_base)
        mu_std, se_std = predict_enn(self._model, self._params, np.asarray([z], dtype=np.float64))
        y_hat = float(self._y_mean + self._y_std * float(mu_std[0]))
        y_se = float(abs(self._y_std) * float(se_std[0]))
        return y_hat, y_se

    def predict_mu(self, *, z_eval: np.ndarray, z_base: np.ndarray, mu0: float, epsilon: float) -> tuple[float, float]:
        y_hat, y_se = self._predict_y(z_eval=z_eval, z_base=z_base)
        if self._target == "delta":
            return float(mu0) + float(epsilon) * float(y_hat), abs(float(epsilon)) * float(y_se)
        return float(y_hat), float(y_se)

    def calibrate_eval(self, *, z_eval: np.ndarray, z_base: np.ndarray, mu_eval_real: float, mu0: float, epsilon: float) -> None:
        if self._model is None or self._params is None:
            return
        try:
            mu_hat, _se_hat = self.predict_mu(z_eval=z_eval, z_base=z_base, mu0=float(mu0), epsilon=float(epsilon))
        except Exception:
            return
        err = abs(float(mu_eval_real) - float(mu_hat))
        if self._abs_err_ema is None:
            self._abs_err_ema = float(err)
        else:
            beta = float(self._cfg.err_ema_beta)
            self._abs_err_ema = beta * float(self._abs_err_ema) + (1.0 - beta) * float(err)
        self._num_calib += 1

    def _should_impute_eval(self) -> bool:
        self._num_candidate_phases += 1
        if len(self._zs) < int(self._cfg.warmup_real_obs):
            return False
        if int(self._cfg.refresh_interval) > 0 and self._num_candidate_phases % int(self._cfg.refresh_interval) == 0:
            return False
        self._maybe_fit()
        if self._model is None or self._params is None:
            return False
        if self._num_calib < int(self._cfg.min_calib_points):
            return False
        if self._abs_err_ema is None:
            return False
        return float(self._abs_err_ema) <= float(self._cfg.max_abs_err_ema)

    def try_impute_eval(self, *, z_eval: np.ndarray, z_base: np.ndarray, mu0: float, epsilon: float) -> tuple[bool, float, float]:
        if not self._should_impute_eval():
            return False, float("nan"), float("nan")
        mu, se = self.predict_mu(z_eval=z_eval, z_base=z_base, mu0=float(mu0), epsilon=float(epsilon))
        if float(se) > float(self._cfg.se_threshold):
            return False, float("nan"), float("nan")
        self._num_imputed += 1
        return True, float(mu), float(se)


def format_enn_stats(imputer, *, label: str = "imputed_minus") -> str:
    err = imputer.abs_err_ema
    err_s = "N/A" if err is None else f"{err:.4f}"
    extra = ""
    num_candidates = int(getattr(getattr(imputer, "_cfg", object()), "num_candidates", 1))
    if hasattr(imputer, "num_selected") and num_candidates > 1:
        extra = f" seedselect={imputer.num_selected}"
    return f"ENN: real_evals={imputer.num_real_evals} {label}={imputer.num_imputed} abs_err_ema={err_s}{extra}"
