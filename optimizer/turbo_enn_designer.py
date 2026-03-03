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
from enn.turbo.optimizer import create_optimizer

import common.all_bounds as all_bounds
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
        use_y_var: bool = False,
        num_candidates: Optional[int] = None,
        candidate_rv: Optional[str] = None,
        num_metrics: Optional[int] = None,
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
        self._use_y_var = use_y_var
        self._num_candidates = num_candidates
        self._candidate_rv = candidate_rv
        self._num_metrics = num_metrics

        self._turbo = None
        self._num_arms = None
        self._rng = np.random.default_rng(np.random.randint(2**31))
        self._num_told = 0
        self._datum_best = None
        self._y_est_best = None

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
        if self._tr_type == "turbo":
            return TurboTRConfig()
        if self._tr_type == "none":
            return NoTRConfig()
        if self._tr_type == "morbo":
            if num_metrics is None:
                raise ValueError("num_metrics is required for tr_type='morbo'")
            from enn.turbo.config.trust_region import MultiObjectiveConfig

            return MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=int(num_metrics)))
        raise ValueError(f"Invalid tr_type: {self._tr_type}")

    def _make_config(self, num_init: int, num_metrics: int | None):
        num_candidates = self._num_candidates
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
