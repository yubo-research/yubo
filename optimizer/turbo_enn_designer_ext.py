from __future__ import annotations

from typing import Optional

import numpy as np
from enn.turbo import turbo_utils
from enn.turbo.config.trust_region import TrustRegionConfig, TurboTRConfig

from optimizer.box_trust_region import maybe_enable_module_masks
from optimizer.submodule_perturbator import leaf_module_param_blocks
from optimizer.trust_region_config import MetricShapedTRConfig, normalize_geometry_name
from optimizer.turbo_enn_designer import TurboENNDesigner as _TurboENNDesigner


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
        f"enabled=True geometry={normalize_geometry_name(geometry)} acq={acq_type} "
        f"params={total_params} blocks={len(block_slices)} expected_blocks={expected_blocks:.2f} "
        f"block_prob={float(block_prob):.3f} block_sizes={block_sizes}"
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
            fixed_length=fixed_length,
            acq_type=acq_type,
            tr_type=tr_type,
            tr_geometry=tr_geometry,
            metric_sampler=metric_sampler,
            metric_rank=metric_rank,
            update_option=update_option,
            use_y_var=use_y_var,
            num_candidates=num_candidates,
            candidate_rv=candidate_rv,
            num_metrics=num_metrics,
            use_python=use_python,
        )
        self._pc_rotation_mode = pc_rotation_mode
        self._pc_rank = pc_rank
        self._tr_length_fixed = tr_length_fixed if tr_length_fixed is not None else fixed_length
        self._fixed_length = self._tr_length_fixed
        self._p_raasp = float(p_raasp)
        self._radial_mode = radial_mode
        self._shape_period = int(shape_period)
        self._shape_ema = float(shape_ema)
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
        if rng is not None:
            self._rng = rng

        if self._tr_type == "turbo":
            _ = MetricShapedTRConfig(
                geometry=normalize_geometry_name(self._tr_geometry),
                metric_sampler=self._metric_sampler,
                metric_rank=self._metric_rank,
                pc_rotation_mode=self._pc_rotation_mode,
                pc_rank=self._pc_rank,
                fixed_length=self._tr_length_fixed,
                update_option=self._update_option,
                p_raasp=self._p_raasp,
                radial_mode=self._radial_mode,
                shape_period=self._shape_period,
                shape_ema=self._shape_ema,
                rho_bad=self._rho_bad,
                rho_good=self._rho_good,
                gamma_down=self._gamma_down,
                gamma_up=self._gamma_up,
                boundary_tol=self._boundary_tol,
                use_accel=self._use_accel,
                accel=self._accel,
            )

    def _make_trust_region(self, num_metrics: int | None) -> TrustRegionConfig:
        if self._tr_type == "turbo":
            geometry = normalize_geometry_name(self._tr_geometry)
            if geometry == "box" and self._metric_sampler is None and self._update_option == "option_a" and self._tr_length_fixed is None:
                return TurboTRConfig()
            kwargs = dict(
                geometry=geometry,
                metric_sampler=self._metric_sampler,
                metric_rank=self._metric_rank,
                pc_rotation_mode=self._pc_rotation_mode,
                pc_rank=self._pc_rank,
                fixed_length=self._tr_length_fixed,
                use_accel=self._use_accel,
                accel=self._accel,
            )
            if geometry in {"enn_ellip", "grad_ellip"}:
                kwargs.update(
                    update_option=self._update_option,
                    p_raasp=self._p_raasp,
                    radial_mode=self._radial_mode,
                    shape_period=self._shape_period,
                    shape_ema=self._shape_ema,
                    rho_bad=self._rho_bad,
                    rho_good=self._rho_good,
                    gamma_down=self._gamma_down,
                    gamma_up=self._gamma_up,
                    boundary_tol=self._boundary_tol,
                )
            return MetricShapedTRConfig(**kwargs)
        return super()._make_trust_region(num_metrics)

    def _init_optimizer(self, data, num_arms):
        super()._init_optimizer(data, num_arms)
        if self._turbo is None:
            return
        tr_state = getattr(self._turbo, "_tr_state", None)
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
