import pickle
from typing import Callable, Optional

import numpy as np
from enn.turbo import turbo_utils
from enn.turbo.config.acq_type import AcqType
from enn.turbo.config.candidate_gen_config import CandidateGenConfig
from enn.turbo.config.candidate_rv import CandidateRV
from enn.turbo.config.enn_surrogate_config import (
    ENNFitConfig,
    ENNSurrogateConfig,
)
from enn.turbo.config.factory import (
    lhd_only_config,
    turbo_enn_config,
    turbo_one_config,
    turbo_zero_config,
)
from enn.turbo.config.raasp_driver import RAASPDriver
from enn.turbo.config.trust_region import (
    MorboTRConfig,
    NoTRConfig,
    TrustRegionConfig,
)

from optimizer.enn_turbo_optimizer import (
    create_optimizer,
)
from optimizer.enn_turbo_optimizer import (
    predict_mu_sigma as turbo_predict_mu_sigma,
)
from optimizer.enn_turbo_optimizer import (
    scalarize as turbo_scalarize,
)
from optimizer.trust_region_config import MetricShapedTRConfig
from optimizer.turbo_enn_runtime import (
    call_designer as _call_designer_runtime,
)
from optimizer.turbo_enn_runtime import (
    get_algo_metrics as _get_algo_metrics_runtime,
)
from optimizer.turbo_enn_runtime import (
    infer_num_metrics as _infer_num_metrics_runtime,
)
from optimizer.turbo_enn_runtime import (
    resolve_num_metrics as _resolve_num_metrics_runtime,
)
from optimizer.turbo_enn_runtime import (
    tell_new_data as _tell_new_data_runtime,
)
from optimizer.turbo_enn_runtime import (
    update_best_estimate as _update_best_estimate_runtime,
)

_X_LOW = -1.0
_X_HIGH = 1.0


def _predict_mu_sigma(opt, x: np.ndarray):
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(1, -1)
    if x_arr.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {x_arr.shape}")
    try:
        bounds = getattr(opt, "_bounds")
        x_unit = turbo_utils.to_unit(x_arr, bounds)
        surrogate = getattr(opt, "_surrogate")
        posterior = surrogate.predict(x_unit)
    except Exception:
        return None
    mu = getattr(posterior, "mu", None)
    sigma = getattr(posterior, "sigma", None)
    if mu is None:
        return None
    mu = np.asarray(mu, dtype=float)
    if mu.ndim == 1:
        mu = mu.reshape(-1, 1)
    if sigma is None:
        sigma = np.zeros_like(mu, dtype=float)
    else:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.ndim == 1:
            sigma = sigma.reshape(-1, 1)
    if mu.shape != sigma.shape:
        raise RuntimeError(f"mu/sigma shape mismatch: {mu.shape} vs {sigma.shape}")
    return mu, sigma


def _validate_tr_options(designer: "TurboENNDesigner") -> None:
    _ = MetricShapedTRConfig(
        geometry=designer._tr_geometry,
        metric_sampler=designer._metric_sampler,
        metric_rank=designer._metric_rank,
        fixed_length=designer._tr_length_fixed,
        update_option=designer._update_option,
        p_raasp=designer._p_raasp,
        radial_mode=designer._radial_mode,
        shape_period=designer._shape_period,
        shape_ema=designer._shape_ema,
        rho_bad=designer._rho_bad,
        rho_good=designer._rho_good,
        gamma_down=designer._gamma_down,
        gamma_up=designer._gamma_up,
        boundary_tol=designer._boundary_tol,
    )


def _as_num_candidates_fn(
    num_candidates: int | Callable[..., int],
) -> Callable[..., int]:
    if callable(num_candidates):
        return num_candidates
    value = int(num_candidates)
    if value <= 0:
        raise ValueError(f"num_candidates must be > 0, got {value}")

    def _fn(*, num_dim: int, num_arms: int) -> int:
        _ = num_dim, num_arms
        return value

    return _fn


def _parse_candidate_rv(candidate_rv: str | None) -> CandidateRV:
    if candidate_rv is None:
        return CandidateRV.SOBOL
    candidate_rv = candidate_rv.lower()
    if candidate_rv == "gpu_uniform":
        candidate_rv = "uniform"
    try:
        return CandidateRV(candidate_rv)
    except ValueError as exc:
        raise ValueError(f"Invalid candidate_rv: {candidate_rv}") from exc


def _parse_acq_type(acq_type: str) -> AcqType:
    try:
        return AcqType(acq_type.lower())
    except ValueError as exc:
        raise ValueError(f"Invalid acq_type: {acq_type}") from exc


def _make_trust_region(designer: "TurboENNDesigner", num_metrics: int | None) -> TrustRegionConfig:
    if designer._tr_type == "turbo":
        return MetricShapedTRConfig(
            geometry=designer._tr_geometry,
            metric_sampler=designer._metric_sampler,
            metric_rank=designer._metric_rank,
            pc_rotation_mode=getattr(designer, "_pc_rotation_mode", None),
            pc_rank=getattr(designer, "_pc_rank", None),
            fixed_length=designer._tr_length_fixed,
            update_option=designer._update_option,
            p_raasp=designer._p_raasp,
            radial_mode=designer._radial_mode,
            shape_period=designer._shape_period,
            shape_ema=designer._shape_ema,
            rho_bad=designer._rho_bad,
            rho_good=designer._rho_good,
            gamma_down=designer._gamma_down,
            gamma_up=designer._gamma_up,
            boundary_tol=designer._boundary_tol,
        )
    if designer._tr_type == "none":
        return NoTRConfig()
    if designer._tr_type == "morbo":
        if num_metrics is None:
            raise ValueError("num_metrics is required for tr_type='morbo'")
        from enn.turbo.config.trust_region import MultiObjectiveConfig

        return MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=int(num_metrics)))
    raise ValueError(f"Invalid tr_type: {designer._tr_type}")


def _make_config(designer: "TurboENNDesigner", num_init: int, num_metrics: int | None):
    num_candidates = designer._num_candidates
    candidate_rv = _parse_candidate_rv(designer._candidate_rv)
    trust_region = _make_trust_region(designer, num_metrics)

    if designer._turbo_mode == "turbo-enn":
        acq_type = _parse_acq_type(designer._acq_type)
        enn = ENNSurrogateConfig(
            k=designer._k,
            fit=ENNFitConfig(
                num_fit_samples=designer._num_fit_samples,
                num_fit_candidates=designer._num_fit_candidates,
            ),
        )
        num_candidates_fn = None if num_candidates is None else _as_num_candidates_fn(num_candidates)
        if num_candidates_fn is not None or designer._candidate_rv is not None:
            candidates = CandidateGenConfig(candidate_rv=candidate_rv, raasp_driver=RAASPDriver.FAST)
            if num_candidates_fn is not None:
                candidates = CandidateGenConfig(
                    candidate_rv=candidate_rv,
                    num_candidates=num_candidates_fn,
                    raasp_driver=RAASPDriver.FAST,
                )
        else:
            candidates = CandidateGenConfig(raasp_driver=RAASPDriver.FAST)
        return turbo_enn_config(
            enn=enn,
            trust_region=trust_region,
            candidates=candidates,
            num_init=num_init,
            trailing_obs=designer._num_keep,
            acq_type=acq_type,
        )
    if designer._turbo_mode == "turbo-zero":
        if callable(num_candidates):
            raise ValueError("num_candidates must be an int for turbo-zero mode")
        return turbo_zero_config(
            num_candidates=num_candidates,
            num_init=num_init,
            trailing_obs=designer._num_keep,
            trust_region=trust_region,
            candidate_rv=candidate_rv,
        )
    if designer._turbo_mode == "turbo-one":
        if callable(num_candidates):
            raise ValueError("num_candidates must be an int for turbo-one mode")
        return turbo_one_config(
            num_candidates=num_candidates,
            num_init=num_init,
            trailing_obs=designer._num_keep,
            trust_region=trust_region,
            candidate_rv=candidate_rv,
        )
    if designer._turbo_mode == "lhd-only":
        if callable(num_candidates):
            raise ValueError("num_candidates must be an int for lhd-only mode")
        return lhd_only_config(
            num_candidates=num_candidates,
            num_init=num_init,
            trailing_obs=designer._num_keep,
            trust_region=trust_region,
            candidate_rv=candidate_rv,
        )
    raise ValueError(f"Invalid turbo mode: {designer._turbo_mode}")


def _validate_turbo_fit_counts(num_fit_samples, num_fit_candidates):
    if num_fit_samples is not None and int(num_fit_samples) <= 0:
        raise ValueError(f"num_fit_samples must be > 0, got {num_fit_samples}")
    if num_fit_candidates is not None and int(num_fit_candidates) <= 0:
        raise ValueError(f"num_fit_candidates must be > 0, got {num_fit_candidates}")


class TurboENNDesigner:
    def __init__(
        self,
        policy,
        turbo_mode: str,
        num_init: Optional[int] = None,
        k: Optional[int] = None,
        num_keep: Optional[int] = None,
        num_fit_samples: Optional[int] = None,
        num_fit_candidates: Optional[int] = None,
        acq_type: str = "pareto",
        tr_type: Optional[str] = None,
        tr_geometry: Optional[str] = None,
        metric_sampler: Optional[str] = None,
        metric_rank: Optional[int] = None,
        pc_rotation_mode: Optional[str] = None,
        pc_rank: Optional[int] = None,
        tr_length_fixed: Optional[float] = None,
        update_option: str = "option_a",
        p_raasp: float = 0.2,
        radial_mode: str = "ball_uniform",
        shape_period: int = 5,
        shape_ema: float = 0.2,
        rho_bad: float = 0.25,
        rho_good: float = 0.75,
        gamma_down: float = 0.5,
        gamma_up: float = 2.0,
        boundary_tol: float = 0.1,
        use_y_var: bool = False,
        num_candidates: Optional[int | Callable[..., int]] = None,
        candidate_rv: Optional[str] = None,
        num_metrics: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self._policy = policy
        if turbo_mode not in ("turbo-enn", "turbo-zero", "turbo-one", "lhd-only"):
            raise ValueError(f"Invalid turbo mode: {turbo_mode}")
        if turbo_mode in ("turbo-zero", "turbo-one", "lhd-only"):
            assert k is None
        self._turbo_mode = turbo_mode
        self._num_init = num_init
        self._k = k
        self._num_keep = num_keep
        self._num_fit_samples = num_fit_samples
        self._num_fit_candidates = num_fit_candidates
        self._acq_type = acq_type
        self._tr_type = tr_type if tr_type is not None else "turbo"
        self._tr_geometry = tr_geometry if tr_geometry is not None else "box"
        self._metric_sampler = metric_sampler
        self._metric_rank = metric_rank
        self._pc_rotation_mode = pc_rotation_mode
        self._pc_rank = pc_rank
        self._tr_length_fixed = tr_length_fixed
        self._update_option = update_option
        self._p_raasp = float(p_raasp)
        self._radial_mode = radial_mode
        self._shape_period = int(shape_period)
        self._shape_ema = float(shape_ema)
        self._rho_bad = float(rho_bad)
        self._rho_good = float(rho_good)
        self._gamma_down = float(gamma_down)
        self._gamma_up = float(gamma_up)
        self._boundary_tol = float(boundary_tol)
        self._use_y_var = use_y_var
        self._num_candidates = num_candidates
        self._candidate_rv = candidate_rv
        self._num_metrics = num_metrics

        _validate_turbo_fit_counts(num_fit_samples, num_fit_candidates)

        self._turbo = None
        self._num_arms = None
        self._rng = rng if rng is not None else np.random.default_rng(np.random.randint(2**31))
        self._num_told = 0
        self._datum_best = None
        self._y_est_best = None

        _validate_tr_options(self)

    def state_dict(self, data=None) -> dict:
        best_index = None
        if data is not None and self._datum_best is not None:
            try:
                best_index = int(data.index(self._datum_best))
            except ValueError:
                best_index = None
        turbo_state = None
        if self._turbo is not None:
            turbo_state = pickle.dumps(self._turbo)
        return {
            "num_arms": self._num_arms,
            "num_told": int(self._num_told),
            "y_est_best": self._y_est_best,
            "datum_best_index": best_index,
            "rng_state": self._rng.bit_generator.state,
            "turbo_state": turbo_state,
        }

    def load_state_dict(self, state: dict, *, data=None) -> None:
        self._num_arms = state.get("num_arms")
        self._num_told = int(state.get("num_told", 0))
        self._y_est_best = state.get("y_est_best")
        turbo_state = state.get("turbo_state")
        if turbo_state is not None:
            self._turbo = pickle.loads(turbo_state)
        else:
            self._turbo = None
        if self._turbo is not None and hasattr(self._turbo, "_rng"):
            self._rng = self._turbo._rng
        rng_state = state.get("rng_state")
        if rng_state is not None:
            try:
                self._rng.bit_generator.state = rng_state
                if self._turbo is not None and hasattr(self._turbo, "_rng"):
                    self._turbo._rng.bit_generator.state = rng_state
            except Exception:
                pass
        best_index = state.get("datum_best_index")
        if best_index is not None and data is not None:
            idx = int(best_index)
            if 0 <= idx < len(data):
                self._datum_best = data[idx]

    def best_datum(self):
        return self._datum_best

    def predict_mu_sigma(self, x: np.ndarray):
        if self._turbo is None:
            return None
        return turbo_predict_mu_sigma(self._turbo, x)

    def scalarize(self, y: np.ndarray, *, clip: bool = False):
        if self._turbo is None:
            return None
        return turbo_scalarize(self._turbo, y, clip=clip)

    def _init_optimizer(self, data, num_arms):
        num_init = (
            self._num_arms
            if self._num_init is None
            else max(
                self._num_arms,
                self._num_arms * int(self._num_init / self._num_arms + 0.5),
            )
        )
        assert num_init > 0 or self._num_init is None
        num_dim = self._policy.num_params()
        bounds = np.array([[_X_LOW, _X_HIGH]] * num_dim)
        num_metrics = self._resolve_num_metrics(data)
        config = _make_config(self, num_init, num_metrics)
        if hasattr(config, "candidates") and config.candidates is not None:
            try:
                n_cand = int(config.candidates.num_candidates(num_dim=num_dim, num_arms=num_arms))
            except Exception:
                n_cand = None
            if n_cand is not None and n_cand < num_arms:
                raise ValueError(f"num_candidates={n_cand} must be >= num_arms={num_arms}")
        self._turbo = create_optimizer(bounds=bounds, config=config, rng=self._rng)

    def _resolve_num_metrics(self, data):
        return _resolve_num_metrics_runtime(self, data)

    def _infer_num_metrics(self, data):
        return _infer_num_metrics_runtime(self, data)

    def _tell_new_data(self, new_data):
        _tell_new_data_runtime(self, new_data)

    def _update_best_estimate(self, new_data, y_est_0):
        _update_best_estimate_runtime(self, new_data, y_est_0)

    def __call__(self, data, num_arms, *, telemetry=None):
        return _call_designer_runtime(self, data, num_arms, telemetry=telemetry)

    def _make_policy(self, x):
        policy = self._policy.clone()
        policy.set_params(x)
        return policy

    def get_algo_metrics(self) -> dict[str, float]:
        return _get_algo_metrics_runtime(self)
