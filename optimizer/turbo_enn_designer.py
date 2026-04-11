from typing import Optional

import numpy as np
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
    TurboTRConfig,
)

import common.all_bounds as all_bounds
from optimizer.designer_asserts import assert_scalar_rreturn
from optimizer.designer_protocol import Designer
from optimizer.trust_region_accel import accel_name as _accel_name
from optimizer.trust_region_config import MetricShapedTRConfig, normalize_geometry_name


def _create_optimizer_auto(bounds, config, rng):
    """Create optimizer using automatic backend selection (Rust preferred)."""
    try:
        from enn import create_optimizer
    except ImportError as exc:
        # Modal/remote images may not have enn_rust; fall back to Python optimizer.
        print(f"[TurboENNDesigner] Rust backend unavailable ({exc}); falling back to Python backend")
        return _create_optimizer_py(bounds=bounds, config=config, rng=rng)

    return create_optimizer(bounds=bounds, config=config, rng=rng)


def _create_optimizer_py(bounds, config, rng):
    """Create optimizer forcing Python backend."""
    from optimizer.enn_turbo_optimizer import create_optimizer

    return create_optimizer(bounds=bounds, config=config, rng=rng)


class TurboENNDesigner(Designer):
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
        update_option: str = "option_a",
        use_y_var: bool = False,
        num_candidates: Optional[int] = None,
        candidate_rv: Optional[str] = None,
        num_metrics: Optional[int] = None,
        use_python: bool = False,
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
        self._fixed_length = fixed_length
        self._acq_type = acq_type
        self._tr_type = tr_type if tr_type is not None else "turbo"
        self._tr_geometry = normalize_geometry_name(tr_geometry if tr_geometry is not None else "box")
        self._metric_sampler = metric_sampler
        self._metric_rank = metric_rank
        self._update_option = update_option
        self._use_y_var = use_y_var
        self._num_candidates = num_candidates
        self._candidate_rv = candidate_rv
        self._num_metrics = num_metrics
        self._use_python = use_python

        self._turbo = None
        self._num_arms = None
        self._rng = np.random.default_rng(np.random.randint(2**31))
        self._num_told = 0
        self._datum_best = None
        self._y_est_best = None

        _validate_tr_options(
            tr_type=self._tr_type,
            tr_geometry=self._tr_geometry,
            metric_sampler=self._metric_sampler,
            metric_rank=self._metric_rank,
            fixed_length=self._fixed_length,
            update_option=self._update_option,
        )

    def _requires_python_backend(self) -> bool:
        if self._use_python:
            return True
        return self._tr_type == "turbo" and self._tr_geometry != "box"

    def _parse_candidate_rv(self) -> CandidateRV:
        if self._candidate_rv is None:
            if self._policy.num_params() >= 10000:
                return CandidateRV.UNIFORM
            return CandidateRV.SOBOL
        candidate_rv = self._candidate_rv.lower()
        if candidate_rv == "gpu_uniform":
            candidate_rv = "uniform"
        try:
            return CandidateRV(candidate_rv)
        except ValueError as exc:
            raise ValueError(f"Invalid candidate_rv: {self._candidate_rv}") from exc

    def _parse_acq_type(self) -> AcqType:
        try:
            return AcqType(self._acq_type.lower())
        except ValueError as exc:
            raise ValueError(f"Invalid acq_type: {self._acq_type}") from exc

    def _make_trust_region(self, num_metrics: int | None) -> TrustRegionConfig:
        return _make_trust_region(
            tr_type=self._tr_type,
            tr_geometry=self._tr_geometry,
            metric_sampler=self._metric_sampler,
            metric_rank=self._metric_rank,
            fixed_length=self._fixed_length,
            update_option=self._update_option,
            num_metrics=num_metrics,
            use_accel=_uses_accel_tr(self._tr_type, self._tr_geometry),
        )

    def _make_config(self, num_init: int, num_metrics: int | None):
        num_candidates = self._num_candidates
        if isinstance(num_candidates, int):
            fixed_num_candidates = int(num_candidates)

            def _fixed_num_candidates(*, num_dim, num_arms):
                _ = num_dim, num_arms
                return fixed_num_candidates

            num_candidates = _fixed_num_candidates
        candidate_rv = self._parse_candidate_rv()
        trust_region = self._make_trust_region(num_metrics)

        if self._turbo_mode == "turbo-enn":
            acq_type = self._parse_acq_type()
            enn = ENNSurrogateConfig(
                k=self._k,
                fit=ENNFitConfig(
                    num_fit_samples=self._num_fit_samples,
                    num_fit_candidates=self._num_fit_candidates,
                ),
            )
            if num_candidates is None:
                candidates = CandidateGenConfig(candidate_rv=candidate_rv, raasp_driver=RAASPDriver.FAST)
            else:
                candidates = CandidateGenConfig(
                    candidate_rv=candidate_rv,
                    num_candidates=num_candidates,
                    raasp_driver=RAASPDriver.FAST,
                )
            return turbo_enn_config(
                enn=enn,
                trust_region=trust_region,
                candidates=candidates,
                num_init=num_init,
                trailing_obs=self._num_keep,
                acq_type=acq_type,
            )
        elif self._turbo_mode == "turbo-zero":
            return turbo_zero_config(
                num_candidates=num_candidates,
                num_init=num_init,
                trailing_obs=self._num_keep,
                trust_region=trust_region,
                candidate_rv=candidate_rv,
            )
        elif self._turbo_mode == "turbo-one":
            return turbo_one_config(
                num_candidates=num_candidates,
                num_init=num_init,
                trailing_obs=self._num_keep,
                trust_region=trust_region,
                candidate_rv=candidate_rv,
            )
        elif self._turbo_mode == "lhd-only":
            return lhd_only_config(
                num_candidates=num_candidates,
                num_init=num_init,
                trailing_obs=self._num_keep,
                trust_region=trust_region,
                candidate_rv=candidate_rv,
            )
        raise ValueError(f"Invalid turbo mode: {self._turbo_mode}")

    def best_datum(self):
        return self._datum_best

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
        create_opt = _create_optimizer_py if self._requires_python_backend() else _create_optimizer_auto
        self._turbo = create_opt(bounds=bounds, config=config, rng=self._rng)
        # Debug: show which backend is being used
        opt_type = type(self._turbo).__name__
        backend = "Rust" if hasattr(self._turbo, "_inner") else "Python"
        print(f"[TurboENNDesigner] Optimizer type: {opt_type} ({backend} backend)")
        tr_state = getattr(self._turbo, "_tr_state", None)
        if _uses_accel_tr(self._tr_type, self._tr_geometry) and tr_state is not None:
            accel_enabled = bool(getattr(tr_state, "use_accel", False))
            accel_kind = _accel_name() if accel_enabled else "none"
            print(f"[TurboENNDesigner] Trust-region accel: enabled={accel_enabled} accel={accel_kind}")

    def _resolve_num_metrics(self, data):
        num_metrics = self._num_metrics
        if self._tr_type != "morbo":
            return num_metrics
        if num_metrics is None:
            num_metrics = self._infer_num_metrics(data)
        if num_metrics < 2:
            raise ValueError("num_metrics must be >= 2 for tr_type='morbo'")
        self._num_metrics = num_metrics
        return num_metrics

    def _infer_num_metrics(self, data):
        policy_metrics = getattr(self._policy, "num_metrics", None)
        if callable(policy_metrics):
            policy_metrics = policy_metrics()
        if policy_metrics is not None:
            return int(policy_metrics)
        if len(data) > 0:
            y = np.asarray([d.trajectory.rreturn for d in data])
            return int(y.shape[1]) if y.ndim == 2 else 1
        return 2

    def _tell_new_data(self, new_data):
        if self._tr_type != "morbo":
            assert_scalar_rreturn(new_data)
        x_list = [d.policy.get_params() for d in new_data]
        y_list = [d.trajectory.rreturn for d in new_data]
        y_se_list = [d.trajectory.rreturn_se for d in new_data] if self._use_y_var else []
        if self._use_y_var:
            assert all(se is not None for se in y_se_list)
        if len(x_list) == 0:
            return
        x = np.array(x_list)
        y_obs = np.array(y_list)
        y_obs = y_obs[:, None] if y_obs.ndim == 1 else y_obs
        y_est = self._turbo.tell(x, y_obs, y_var=np.array(y_se_list) ** 2) if y_se_list else self._turbo.tell(x, y_obs)
        assert y_obs.shape == y_est.shape and y_obs.shape[0] == len(new_data)
        if y_est.shape[1] == 1:
            self._update_best_estimate(new_data, y_est[:, 0])

    def _update_best_estimate(self, new_data, y_est_0):
        y_est_0 = np.asarray(y_est_0, dtype=np.float64)
        for i, d in enumerate(new_data):
            d.trajectory.rreturn_est = float(y_est_0[i])
        best_i = int(np.argmax(y_est_0))
        best_y = float(y_est_0[best_i])
        if self._y_est_best is None or best_y > float(self._y_est_best):
            self._y_est_best = best_y
            self._datum_best = new_data[best_i]

    def __call__(self, data, num_arms, *, telemetry=None):
        if self._num_arms is None:
            self._num_arms = num_arms
            self._init_optimizer(data, num_arms)

        if len(data) > self._num_told:
            self._tell_new_data(data[self._num_told :])
            self._num_told = len(data)

        x_new = self._turbo.ask(num_arms)
        if telemetry is not None:
            t = self._turbo.telemetry()
            telemetry.set_dt_fit(t.dt_fit)
            telemetry.set_dt_select(t.dt_sel)

        return [self._make_policy(x) for x in x_new]

    def _make_policy(self, x):
        policy = self._policy.clone()
        policy.set_params(x)
        setattr(policy, "_turbo_enn_eval_reuse_ok", True)
        return policy


def _validate_tr_options(
    *,
    tr_type: str,
    tr_geometry: str,
    metric_sampler: Optional[str],
    metric_rank: Optional[int],
    fixed_length: Optional[float],
    update_option: str,
) -> None:
    if tr_type != "turbo":
        return
    _ = MetricShapedTRConfig(
        geometry=tr_geometry,
        metric_sampler=metric_sampler,
        metric_rank=metric_rank,
        fixed_length=fixed_length,
        update_option=update_option,
    )


def _make_trust_region(
    *,
    tr_type: str,
    tr_geometry: str,
    metric_sampler: Optional[str],
    metric_rank: Optional[int],
    fixed_length: Optional[float],
    update_option: str,
    num_metrics: int | None,
    use_accel: bool,
) -> TrustRegionConfig:
    if tr_type == "turbo":
        if tr_geometry == "box" and metric_sampler is None and update_option == "option_a" and fixed_length is None:
            return TurboTRConfig()
        kwargs = {
            "geometry": tr_geometry,
            "metric_sampler": metric_sampler,
            "metric_rank": metric_rank,
            "fixed_length": fixed_length,
            "use_accel": use_accel,
        }
        if tr_geometry in {"enn_ellip", "grad_ellip"}:
            kwargs["update_option"] = update_option
        return MetricShapedTRConfig(**kwargs)
    if tr_type == "none":
        return NoTRConfig()
    if tr_type == "morbo":
        if num_metrics is None:
            raise ValueError("num_metrics is required for tr_type='morbo'")
        from enn.turbo.config.trust_region import MultiObjectiveConfig

        return MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=int(num_metrics)))
    raise ValueError(f"Invalid tr_type: {tr_type}")


def _uses_accel_tr(tr_type: str, tr_geometry: str) -> bool:
    return tr_type == "turbo" and tr_geometry != "box"
