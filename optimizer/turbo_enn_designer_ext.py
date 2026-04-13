from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from enn.turbo import turbo_utils
from enn.turbo.config.trust_region import NoTRConfig, TrustRegionConfig, TurboTRConfig

import common.all_bounds as all_bounds
from optimizer.box_trust_region import maybe_enable_module_masks
from optimizer.submodule_perturbator import leaf_module_param_blocks
from optimizer.trust_region_accel import accel_name as _accel_name
from optimizer.trust_region_config import MetricShapedTRConfig
from optimizer.turbo_enn_designer import TurboENNDesigner as _TurboENNDesigner
from optimizer.turbo_enn_designer import _create_optimizer_auto


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


def _create_optimizer_py_local(bounds, config, rng):
    from optimizer.enn_turbo_optimizer import create_optimizer

    return create_optimizer(bounds=bounds, config=config, rng=rng)


def _module_tr_summary(
    policy: object,
    block_slices: tuple[tuple[int, int], ...],
    *,
    block_prob: float,
    geometry: str,
    acq_type: str,
) -> str:
    num_params = getattr(policy, "num_params", None)
    total_params = int(num_params()) if callable(num_params) else 0
    block_sizes = [int(end) - int(start) for start, end in block_slices]
    expected_blocks = max(1.0, float(len(block_slices)) * float(block_prob))
    return (
        "MODULE_TR "
        f"enabled=True geometry={str(geometry).strip()} acq={acq_type} "
        f"params={total_params} blocks={len(block_slices)} expected_blocks={expected_blocks:.2f} "
        f"block_prob={float(block_prob):.3f} block_sizes={block_sizes}"
    )


@dataclass(frozen=True)
class _TrustRegionSpec:
    tr_type: str
    tr_geometry: str
    covmat: Optional[str]
    metric_rank: Optional[int]
    fixed_length: Optional[float]
    update_option: str
    p_raasp: float
    radial_mode: str
    shape_period: int
    shape_ema: float
    shape_jitter: float
    shape_kappa_max: float
    rho_bad: float
    rho_good: float
    gamma_down: float
    gamma_up: float
    boundary_tol: float
    use_accel: bool
    accel: Optional[str]


def _validate_tr_options(spec: _TrustRegionSpec) -> None:
    if spec.tr_type != "turbo":
        return
    _ = MetricShapedTRConfig(
        geometry=spec.tr_geometry,
        covmat=spec.covmat,
        metric_rank=spec.metric_rank,
        fixed_length=spec.fixed_length,
        update_option=spec.update_option,
        p_raasp=spec.p_raasp,
        radial_mode=spec.radial_mode,
        shape_period=spec.shape_period,
        shape_ema=spec.shape_ema,
        shape_jitter=spec.shape_jitter,
        shape_kappa_max=spec.shape_kappa_max,
        rho_bad=spec.rho_bad,
        rho_good=spec.rho_good,
        gamma_down=spec.gamma_down,
        gamma_up=spec.gamma_up,
        boundary_tol=spec.boundary_tol,
        use_accel=spec.use_accel,
        accel=spec.accel,
    )


def _make_trust_region(spec: _TrustRegionSpec, *, num_metrics: int | None) -> TrustRegionConfig:
    if spec.tr_type == "turbo":
        if spec.tr_geometry == "box" and spec.covmat is None and spec.update_option == "option_a" and spec.fixed_length is None:
            return TurboTRConfig()
        kwargs = {
            "geometry": spec.tr_geometry,
            "covmat": spec.covmat,
            "metric_rank": spec.metric_rank,
            "fixed_length": spec.fixed_length,
            "use_accel": spec.use_accel,
            "accel": spec.accel,
        }
        if spec.tr_geometry in {"enn_ellip", "grad_ellip"}:
            kwargs.update(
                update_option=spec.update_option,
                p_raasp=spec.p_raasp,
                radial_mode=spec.radial_mode,
                shape_period=spec.shape_period,
                shape_ema=spec.shape_ema,
                shape_jitter=spec.shape_jitter,
                shape_kappa_max=spec.shape_kappa_max,
                rho_bad=spec.rho_bad,
                rho_good=spec.rho_good,
                gamma_down=spec.gamma_down,
                gamma_up=spec.gamma_up,
                boundary_tol=spec.boundary_tol,
            )
        return MetricShapedTRConfig(**kwargs)
    if spec.tr_type == "none":
        return NoTRConfig()
    if spec.tr_type == "morbo":
        if num_metrics is None:
            raise ValueError("num_metrics is required for tr_type='morbo'")
        from enn.turbo.config.trust_region import MorboTRConfig, MultiObjectiveConfig

        return MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=int(num_metrics)))
    raise ValueError(f"Invalid tr_type: {spec.tr_type}")


def _uses_accel_tr(tr_type: str, tr_geometry: str) -> bool:
    return tr_type == "turbo" and tr_geometry != "box"


def _uses_custom_tr(spec: _TrustRegionSpec) -> bool:
    return not (
        spec.tr_type == "turbo" and spec.tr_geometry == "box" and spec.covmat is None and spec.update_option == "option_a" and spec.fixed_length is None
    )


class TurboENNDesigner(_TurboENNDesigner):
    def __init__(
        self,
        policy,
        turbo_mode: str,
        num_init: Optional[int] = None,
        k: Optional[int] = None,
        num_keep: Optional[int] = None,
        num_fit_samples: Optional[int] = None,
        num_fit_candidates: Optional[int] = None,
        fixed_length: Optional[float] = None,
        acq_type: str = "pareto",
        tr_type: Optional[str] = None,
        tr_geometry: Optional[str] = None,
        covmat: Optional[str] = None,
        metric_rank: Optional[int] = None,
        tr_length_fixed: Optional[float] = None,
        update_option: str = "option_a",
        p_raasp: float = 0.2,
        radial_mode: str = "ball_uniform",
        shape_period: int = 5,
        shape_ema: float = 0.2,
        shape_jitter: float = 1e-6,
        shape_kappa_max: float = 1e4,
        rho_bad: float = 0.25,
        rho_good: float = 0.75,
        gamma_down: float = 0.5,
        gamma_up: float = 2.0,
        boundary_tol: float = 0.1,
        use_y_var: bool = False,
        num_candidates: Optional[int] = None,
        candidate_rv: Optional[str] = None,
        num_metrics: Optional[int] = None,
        use_python: bool = False,
        use_accel: bool | None = None,
        accel: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
        module_tr: bool = False,
        module_tr_block_prob: float = 0.5,
        module_tr_min_num_params: int = 10000,
    ):
        super().__init__(
            policy,
            turbo_mode=turbo_mode,
            num_init=num_init,
            k=k,
            num_keep=num_keep,
            num_fit_samples=num_fit_samples,
            num_fit_candidates=num_fit_candidates,
            acq_type=acq_type,
            tr_type=tr_type,
            use_y_var=use_y_var,
            num_candidates=num_candidates,
            candidate_rv=candidate_rv,
            num_metrics=num_metrics,
            use_python=use_python,
        )
        self._tr_geometry = str(tr_geometry if tr_geometry is not None else "box").strip()
        self._covmat = covmat
        self._metric_rank = metric_rank
        self._tr_length_fixed = tr_length_fixed if tr_length_fixed is not None else fixed_length
        self._fixed_length = self._tr_length_fixed
        self._update_option = update_option
        self._p_raasp = float(p_raasp)
        self._radial_mode = radial_mode
        self._shape_period = int(shape_period)
        self._shape_ema = float(shape_ema)
        self._shape_jitter = float(shape_jitter)
        self._shape_kappa_max = float(shape_kappa_max)
        self._rho_bad = float(rho_bad)
        self._rho_good = float(rho_good)
        self._gamma_down = float(gamma_down)
        self._gamma_up = float(gamma_up)
        self._boundary_tol = float(boundary_tol)
        if use_accel is None:
            use_accel = self._tr_type == "turbo" and self._tr_geometry != "box"
        self._use_accel = bool(use_accel)
        self._accel = accel
        self._module_tr = bool(module_tr)
        self._module_tr_block_prob = float(module_tr_block_prob)
        self._module_tr_min_num_params = int(module_tr_min_num_params)
        self._use_python = bool(self._use_python or _uses_accel_tr(self._tr_type, self._tr_geometry))
        if rng is not None:
            self._rng = rng
        self._tr_spec = _TrustRegionSpec(
            tr_type=self._tr_type,
            tr_geometry=self._tr_geometry,
            covmat=self._covmat,
            metric_rank=self._metric_rank,
            fixed_length=self._tr_length_fixed,
            update_option=self._update_option,
            p_raasp=self._p_raasp,
            radial_mode=self._radial_mode,
            shape_period=self._shape_period,
            shape_ema=self._shape_ema,
            shape_jitter=self._shape_jitter,
            shape_kappa_max=self._shape_kappa_max,
            rho_bad=self._rho_bad,
            rho_good=self._rho_good,
            gamma_down=self._gamma_down,
            gamma_up=self._gamma_up,
            boundary_tol=self._boundary_tol,
            use_accel=self._use_accel,
            accel=self._accel,
        )
        _validate_tr_options(self._tr_spec)

    def _make_trust_region(self, num_metrics: int | None) -> TrustRegionConfig:
        return _make_trust_region(self._tr_spec, num_metrics=num_metrics)

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
        bounds = np.array([[all_bounds.x_low, all_bounds.x_high]] * num_dim)
        num_metrics = self._resolve_num_metrics(data)
        config = self._make_config(num_init, num_metrics)
        if self._use_python or _uses_custom_tr(self._tr_spec):
            self._turbo = _create_optimizer_py_local(bounds=bounds, config=config, rng=self._rng)
        else:
            self._turbo = _create_optimizer_auto(bounds=bounds, config=config, rng=self._rng)
        opt_type = type(self._turbo).__name__
        backend = "Rust" if hasattr(self._turbo, "_inner") else "Python"
        print(f"[TurboENNDesigner] Optimizer type: {opt_type} ({backend} backend)")
        tr_state = getattr(self._turbo, "_tr_state", None)
        if _uses_accel_tr(self._tr_type, self._tr_geometry) and tr_state is not None:
            accel_enabled = bool(getattr(tr_state, "use_accel", False))
            accel_kind = _accel_name() if accel_enabled else "none"
            print(f"[TurboENNDesigner] Trust-region accel: enabled={accel_enabled} accel={accel_kind}")
        block_slices: tuple[tuple[int, int], ...] = ()
        if tr_state is not None and self._module_tr and hasattr(self._policy, "parameters"):
            try:
                block_slices = leaf_module_param_blocks(self._policy)
            except Exception:
                block_slices = ()
            setattr(tr_state, "module_block_slices", block_slices)
            setattr(tr_state, "module_block_prob", self._module_tr_block_prob)
        enabled = maybe_enable_module_masks(
            self._turbo,
            self._policy,
            enabled=self._module_tr,
            min_num_params=self._module_tr_min_num_params,
            block_prob=self._module_tr_block_prob,
        )
        if enabled and block_slices:
            print(
                _module_tr_summary(
                    self._policy,
                    block_slices,
                    block_prob=self._module_tr_block_prob,
                    geometry=self._tr_geometry,
                    acq_type=self._acq_type,
                ),
                flush=True,
            )

    def predict_mu_sigma(self, x: np.ndarray):
        if self._turbo is None:
            return None
        return _predict_mu_sigma(self._turbo, x)

    def scalarize(self, y: np.ndarray, *, clip: bool = False):
        if self._turbo is None:
            return None
        tr_state = getattr(self._turbo, "_tr_state", None)
        if tr_state is None or not hasattr(tr_state, "scalarize"):
            return None
        try:
            return tr_state.scalarize(y, clip=clip)
        except Exception:
            return None


__all__ = ["TurboENNDesigner"]
