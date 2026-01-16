from typing import Optional

import numpy as np

import common.all_bounds as all_bounds
from optimizer.designer_asserts import assert_scalar_rreturn
from third_party.enn.turbo import Turbo, TurboMode
from third_party.enn.turbo.turbo_config import TurboConfig, TurboENNConfig, TurboOneConfig, TurboZeroConfig


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
    ):
        self._policy = policy
        if turbo_mode == "turbo-enn":
            self._turbo_mode = TurboMode.TURBO_ENN
        elif turbo_mode == "turbo-zero":
            self._turbo_mode = TurboMode.TURBO_ZERO
            assert k is None
        elif turbo_mode == "turbo-one":
            self._turbo_mode = TurboMode.TURBO_ONE
            assert k is None
        else:
            raise ValueError(f"Invalid turbo mode: {turbo_mode}")
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

        self._turbo = None
        self._num_arms = None
        self._rng = np.random.default_rng(np.random.randint(2**31))
        self._num_told = 0

    def _make_config(self, num_init: int) -> TurboConfig:
        num_candidates = self._num_candidates
        candidate_rv = self._candidate_rv if self._candidate_rv is not None else "sobol"

        if self._turbo_mode == TurboMode.TURBO_ENN:
            return TurboENNConfig(
                k=self._k,
                num_init=num_init,
                trailing_obs=self._num_keep,
                num_fit_samples=self._num_fit_samples,
                num_fit_candidates=self._num_fit_candidates,
                acq_type=self._acq_type,
                tr_type=self._tr_type,
                candidate_rv=candidate_rv,
                num_candidates=num_candidates,
            )
        elif self._turbo_mode == TurboMode.TURBO_ZERO:
            return TurboZeroConfig(
                num_init=num_init,
                trailing_obs=self._num_keep,
                tr_type=self._tr_type,
                candidate_rv=candidate_rv,
                num_candidates=num_candidates,
            )
        elif self._turbo_mode == TurboMode.TURBO_ONE:
            return TurboOneConfig(
                num_init=num_init,
                trailing_obs=self._num_keep,
                tr_type=self._tr_type,
                candidate_rv=candidate_rv,
                num_candidates=num_candidates,
            )
        else:
            raise ValueError(f"Invalid turbo mode: {self._turbo_mode}")

    def __call__(self, data, num_arms, *, telemetry=None):
        TurboENNConfig(
            k=10,
            num_candidates=None,
            num_init=1,
            trailing_obs=None,
            tr_type="turbo",
            num_metrics=None,
            candidate_rv="sobol",
            acq_type="ucb",
            num_fit_samples=100,
            num_fit_candidates=100,
            scale_x=False,
        )
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
            config = self._make_config(num_init)

            self._turbo = Turbo(
                bounds=bounds,
                mode=self._turbo_mode,
                rng=self._rng,
                config=config,
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
            assert len(y_se_list) == 0 or len(y_se_list) == len(y_list), (len(y_se_list), len(y_list))
            if len(x_list) > 0:
                x = np.array(x_list)
                y = np.array(y_list)
                if len(y_se_list) > 0:
                    y_se = np.array(y_se_list)
                    # print("Using y_var", y_se)
                    self._turbo.tell(x, y, y_var=y_se**2)
                else:
                    self._turbo.tell(x, y)
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
