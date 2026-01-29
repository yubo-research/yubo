from typing import Optional

import numpy as np

import common.all_bounds as all_bounds
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
from optimizer.designer_asserts import assert_scalar_rreturn


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
            candidates = None
            if num_candidates is not None or self._candidate_rv is not None:
                if num_candidates is None:
                    candidates = CandidateGenConfig(candidate_rv=candidate_rv, raasp_driver=RAASPDriver.FAST)
                else:
                    candidates = CandidateGenConfig(
                        candidate_rv=candidate_rv,
                        num_candidates=num_candidates,
                        raasp_driver=RAASPDriver.FAST,
                    )
            else:
                candidates = CandidateGenConfig(raasp_driver=RAASPDriver.FAST)
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

    def __call__(self, data, num_arms, *, telemetry=None):
        if self._num_arms is None:
            self._num_arms = num_arms
            if self._num_init is not None:
                num_init = max(self._num_arms, self._num_init)
                num_init = self._num_arms * int(num_init / self._num_arms + 0.5)
                assert num_init > 0, (num_init, self._num_init, self._num_arms)
            else:
                num_init = self._num_arms
            num_dim = self._policy.num_params()
            bounds = np.array([[all_bounds.x_low, all_bounds.x_high]] * num_dim)
            num_metrics = self._num_metrics
            if self._tr_type == "morbo":
                if num_metrics is None:
                    policy_metrics = getattr(self._policy, "num_metrics", None)
                    if callable(policy_metrics):
                        policy_metrics = policy_metrics()
                    if policy_metrics is not None:
                        num_metrics = int(policy_metrics)
                    elif len(data) > 0:
                        y = np.asarray([d.trajectory.rreturn for d in data])
                        num_metrics = int(y.shape[1]) if y.ndim == 2 else 1
                    else:
                        num_metrics = 2
                if num_metrics < 2:
                    raise ValueError("num_metrics must be >= 2 for tr_type='morbo'")
                self._num_metrics = num_metrics
            config = self._make_config(num_init, num_metrics)

            self._turbo = create_optimizer(
                bounds=bounds,
                config=config,
                rng=self._rng,
            )

        if len(data) > self._num_told:
            new_data = data[self._num_told :]
            if self._tr_type != "morbo":
                assert_scalar_rreturn(new_data)
            x_list = []
            y_list = []
            y_se_list = []
            for d in new_data:
                x_list.append(d.policy.get_params())
                y_list.append(d.trajectory.rreturn)
            if self._use_y_var:
                for d in new_data:
                    assert d.trajectory.rreturn_se is not None
                    y_se_list.append(d.trajectory.rreturn_se)
            assert len(y_se_list) == 0 or len(y_se_list) == len(y_list), (
                len(y_se_list),
                len(y_list),
            )
            if len(x_list) > 0:
                x = np.array(x_list)
                y_obs = np.array(y_list)
                if len(y_obs.shape) == 1:
                    y_obs = y_obs[:, None]
                if len(y_se_list) > 0:
                    y_se = np.array(y_se_list)
                    # print("Using y_var", y_se)
                    y_est = self._turbo.tell(x, y_obs, y_var=y_se**2)
                else:
                    y_est = self._turbo.tell(x, y_obs)
                assert y_obs.shape == y_est.shape, (y_obs.shape, y_est.shape)
                assert y_obs.shape[0] == len(new_data), (y_obs.shape, len(new_data))
                if y_est.shape[1] == 1:
                    y_est_0 = np.asarray(y_est[:, 0], dtype=np.float64)
                    for i, d in enumerate(new_data):
                        d.trajectory.rreturn_est = float(y_est_0[i])
                    best_i = int(np.argmax(y_est_0))
                    best_y = float(y_est_0[best_i])
                    if self._y_est_best is None or best_y > float(self._y_est_best):
                        self._y_est_best = best_y
                        self._datum_best = new_data[best_i]

            self._num_told = len(data)

        x_new = self._turbo.ask(num_arms)
        turbo_telemetry = self._turbo.telemetry()
        if telemetry is not None:
            telemetry.set_dt_fit(turbo_telemetry.dt_fit)
            telemetry.set_dt_select(turbo_telemetry.dt_sel)
        policies = []
        for x in x_new:
            policy = self._policy.clone()
            policy.set_params(x)
            policies.append(policy)
        return policies
