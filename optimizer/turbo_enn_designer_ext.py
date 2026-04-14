from __future__ import annotations

from dataclasses import dataclass, field
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
class TurboENNTrustRegionSpec:
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


@dataclass(frozen=True)
class ModuleTRSpec:
    enabled: bool = False
    block_prob: float = 0.5
    min_num_params: int = 10000


@dataclass(frozen=True)
class TurboENNExtConfig:
    turbo_mode: str
    num_init: Optional[int] = None
    k: Optional[int] = None
    num_keep: Optional[int] = None
    num_fit_samples: Optional[int] = None
    num_fit_candidates: Optional[int] = None
    acq_type: str = "pareto"
    trust_region: TurboENNTrustRegionSpec = field(
        default_factory=lambda: TurboENNTrustRegionSpec(
            tr_type="turbo",
            tr_geometry="box",
            covmat=None,
            metric_rank=None,
            fixed_length=None,
            update_option="option_a",
            p_raasp=0.2,
            radial_mode="ball_uniform",
            shape_period=5,
            shape_ema=0.2,
            shape_jitter=1e-6,
            shape_kappa_max=1e4,
            rho_bad=0.25,
            rho_good=0.75,
            gamma_down=0.5,
            gamma_up=2.0,
            boundary_tol=0.1,
            use_accel=False,
            accel=None,
        )
    )
    use_y_var: bool = False
    num_candidates: Optional[int] = None
    candidate_rv: Optional[str] = None
    num_metrics: Optional[int] = None
    use_python: bool = False
    module_tr: ModuleTRSpec = field(default_factory=ModuleTRSpec)
    rng: Optional[np.random.Generator] = None


def _validate_tr_options(spec: TurboENNTrustRegionSpec) -> None:
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


def _make_trust_region(spec: TurboENNTrustRegionSpec, *, num_metrics: int | None) -> TrustRegionConfig:
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


def _uses_custom_tr(spec: TurboENNTrustRegionSpec) -> bool:
    return not (
        spec.tr_type == "turbo" and spec.tr_geometry == "box" and spec.covmat is None and spec.update_option == "option_a" and spec.fixed_length is None
    )


def _config_from_legacy_kwargs(kwargs: dict) -> TurboENNExtConfig:
    data = dict(kwargs)
    tr_type = data.pop("tr_type", None) or "turbo"
    tr_geometry = data.pop("tr_geometry", None) or "box"
    use_accel = data.pop("use_accel", None)
    trust_region = TurboENNTrustRegionSpec(
        tr_type=tr_type,
        tr_geometry=tr_geometry,
        covmat=data.pop("covmat", None),
        metric_rank=data.pop("metric_rank", None),
        fixed_length=data.pop("tr_length_fixed", data.pop("fixed_length", None)),
        update_option=data.pop("update_option", "option_a"),
        p_raasp=float(data.pop("p_raasp", 0.2)),
        radial_mode=data.pop("radial_mode", "ball_uniform"),
        shape_period=int(data.pop("shape_period", 5)),
        shape_ema=float(data.pop("shape_ema", 0.2)),
        shape_jitter=float(data.pop("shape_jitter", 1e-6)),
        shape_kappa_max=float(data.pop("shape_kappa_max", 1e4)),
        rho_bad=float(data.pop("rho_bad", 0.25)),
        rho_good=float(data.pop("rho_good", 0.75)),
        gamma_down=float(data.pop("gamma_down", 0.5)),
        gamma_up=float(data.pop("gamma_up", 2.0)),
        boundary_tol=float(data.pop("boundary_tol", 0.1)),
        use_accel=bool(tr_type == "turbo" and tr_geometry != "box") if use_accel is None else bool(use_accel),
        accel=data.pop("accel", None),
    )
    module_tr = ModuleTRSpec(
        enabled=bool(data.pop("module_tr", False)),
        block_prob=float(data.pop("module_tr_block_prob", 0.5)),
        min_num_params=int(data.pop("module_tr_min_num_params", 10000)),
    )
    config = TurboENNExtConfig(
        turbo_mode=data.pop("turbo_mode"),
        num_init=data.pop("num_init", None),
        k=data.pop("k", None),
        num_keep=data.pop("num_keep", None),
        num_fit_samples=data.pop("num_fit_samples", None),
        num_fit_candidates=data.pop("num_fit_candidates", None),
        acq_type=data.pop("acq_type", "pareto"),
        trust_region=trust_region,
        use_y_var=bool(data.pop("use_y_var", False)),
        num_candidates=data.pop("num_candidates", None),
        candidate_rv=data.pop("candidate_rv", None),
        num_metrics=data.pop("num_metrics", None),
        use_python=bool(data.pop("use_python", False)),
        module_tr=module_tr,
        rng=data.pop("rng", None),
    )
    if data:
        keys = ", ".join(sorted(data))
        raise TypeError(f"Unexpected TurboENNDesigner keyword(s): {keys}")
    return config


class TurboENNDesigner(_TurboENNDesigner):
    def __init__(
        self,
        policy,
        *,
        config: TurboENNExtConfig | None = None,
        **legacy_kwargs,
    ):
        if config is None:
            config = _config_from_legacy_kwargs(legacy_kwargs)
        elif legacy_kwargs:
            keys = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"Unexpected TurboENNDesigner keyword(s) with config=: {keys}")
        trust_region = config.trust_region
        module = config.module_tr
        super().__init__(
            policy,
            turbo_mode=config.turbo_mode,
            num_init=config.num_init,
            k=config.k,
            num_keep=config.num_keep,
            num_fit_samples=config.num_fit_samples,
            num_fit_candidates=config.num_fit_candidates,
            acq_type=config.acq_type,
            tr_type=trust_region.tr_type,
            use_y_var=config.use_y_var,
            num_candidates=config.num_candidates,
            candidate_rv=config.candidate_rv,
            num_metrics=config.num_metrics,
            use_python=config.use_python,
        )
        self._tr_geometry = str(trust_region.tr_geometry).strip()
        self._covmat = trust_region.covmat
        self._metric_rank = trust_region.metric_rank
        self._tr_length_fixed = trust_region.fixed_length
        self._fixed_length = self._tr_length_fixed
        self._update_option = trust_region.update_option
        self._p_raasp = float(trust_region.p_raasp)
        self._radial_mode = trust_region.radial_mode
        self._shape_period = int(trust_region.shape_period)
        self._shape_ema = float(trust_region.shape_ema)
        self._shape_jitter = float(trust_region.shape_jitter)
        self._shape_kappa_max = float(trust_region.shape_kappa_max)
        self._rho_bad = float(trust_region.rho_bad)
        self._rho_good = float(trust_region.rho_good)
        self._gamma_down = float(trust_region.gamma_down)
        self._gamma_up = float(trust_region.gamma_up)
        self._boundary_tol = float(trust_region.boundary_tol)
        self._use_accel = bool(trust_region.use_accel)
        self._accel = trust_region.accel
        self._module_tr = bool(module.enabled)
        self._module_tr_block_prob = float(module.block_prob)
        self._module_tr_min_num_params = int(module.min_num_params)
        self._use_python = bool(self._use_python or _uses_accel_tr(self._tr_type, self._tr_geometry))
        if config.rng is not None:
            self._rng = config.rng
        self._tr_spec = trust_region
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


__all__ = [
    "TurboENNDesigner",
    "TurboENNTrustRegionSpec",
    "ModuleTRSpec",
    "TurboENNExtConfig",
]
