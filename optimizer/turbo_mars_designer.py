from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .mars_config import BayesianMarsSurrogateConfig, MarsSurrogateConfig
from .mars_enn_config import MarsENNSurrogateConfig
from .mars_turbo_optimizer import create_mars_optimizer
from .turbo_enn_designer import TurboENNDesigner
from .turbo_mars_config import TurboBayesianMARSDesignerConfig, TurboMARSDesignerConfig, TurboMarsENNDesignerConfig

_logger = logging.getLogger(__name__)


class TurboMARSDesigner(TurboENNDesigner):
    def __init__(self, policy: Any, *, config: TurboMARSDesignerConfig) -> None:
        self._mars_config = config.mars
        super().__init__(
            policy,
            turbo_mode="turbo-enn",
            k=1,
            num_init=config.num_init,
            num_keep=config.num_keep,
            num_fit_samples=1,
            num_fit_candidates=1,
            acq_type=config.acq_type,
            tr_type=config.tr_type,
            use_y_var=False,
            num_candidates=config.num_candidates,
            candidate_rv=config.candidate_rv,
            use_python=False,
        )

    def _make_config(self, num_init: int, num_metrics: int | None):
        _check_scalar_metrics(num_metrics, "TurboMARSDesigner")
        return _optimizer_config(
            self,
            num_init=num_init,
            surrogate=self._mars_config,
        )

    def _init_optimizer(self, data, num_arms):
        _init_mars_optimizer(self, data, num_arms)


class TurboBayesianMARSDesigner(TurboENNDesigner):
    def __init__(self, policy: Any, *, config: TurboBayesianMARSDesignerConfig) -> None:
        self._bmars_config = config.bmars
        self._bmars_pass_y_var: bool | None = None
        super().__init__(
            policy,
            turbo_mode="turbo-enn",
            k=1,
            num_init=config.num_init,
            num_keep=config.num_keep,
            num_fit_samples=1,
            num_fit_candidates=1,
            acq_type=config.acq_type,
            tr_type=config.tr_type,
            use_y_var=True,
            num_candidates=config.num_candidates,
            candidate_rv=config.candidate_rv,
            use_python=False,
        )

    def _make_config(self, num_init: int, num_metrics: int | None):
        _check_scalar_metrics(num_metrics, "TurboBayesianMARSDesigner")
        return _optimizer_config(
            self,
            num_init=num_init,
            surrogate=self._bmars_config,
        )

    def _init_optimizer(self, data, num_arms):
        _init_mars_optimizer(self, data, num_arms)

    def _tell_new_data(self, new_data):
        _tell_bmars_new_data(self, new_data)


class TurboMarsENNDesigner(TurboENNDesigner):
    def __init__(self, policy: Any, *, config: TurboMarsENNDesignerConfig) -> None:
        self._mars_enn_config = config.mars_enn
        super().__init__(
            policy,
            turbo_mode="turbo-enn",
            k=1,
            num_init=config.num_init,
            num_keep=config.num_keep,
            num_fit_samples=1,
            num_fit_candidates=1,
            acq_type=config.acq_type,
            tr_type=config.tr_type,
            use_y_var=False,
            num_candidates=config.num_candidates,
            candidate_rv=config.candidate_rv,
            use_python=False,
        )

    def _make_config(self, num_init: int, num_metrics: int | None):
        _check_scalar_metrics(num_metrics, "TurboMarsENNDesigner")
        return _optimizer_config(
            self,
            num_init=num_init,
            surrogate=self._mars_enn_config,
        )

    def _init_optimizer(self, data, num_arms):
        _init_mars_optimizer(self, data, num_arms)


def _optimizer_config(
    designer: TurboENNDesigner,
    *,
    num_init: int,
    surrogate: MarsSurrogateConfig | BayesianMarsSurrogateConfig | MarsENNSurrogateConfig,
):
    from enn.turbo.config.candidate_gen_config import CandidateGenConfig
    from enn.turbo.config.init_config import InitConfig
    from enn.turbo.config.observation_history_config import ObservationHistoryConfig
    from enn.turbo.config.optimizer_config import OptimizerConfig
    from enn.turbo.config.raasp_driver import RAASPDriver

    acquisition, acq_optimizer = _acq_config(designer._acq_type)
    return OptimizerConfig(
        trust_region=designer._make_trust_region(None),
        candidates=CandidateGenConfig(
            candidate_rv=designer._parse_candidate_rv(),
            num_candidates=designer._num_candidates,
            raasp_driver=RAASPDriver.FAST,
        ),
        init=InitConfig(num_init=num_init),
        surrogate=surrogate,
        acquisition=acquisition,
        acq_optimizer=acq_optimizer,
        observation_history=ObservationHistoryConfig(),
    )


def _acq_config(acq_type: str):
    from enn.turbo.config.acq_type import AcqType
    from enn.turbo.config.acquisition import (
        DrawAcquisitionConfig,
        NDSOptimizerConfig,
        ParetoAcquisitionConfig,
        RAASPOptimizerConfig,
        UCBAcquisitionConfig,
    )

    acq = AcqType(str(acq_type).lower())
    if acq == AcqType.PARETO:
        return ParetoAcquisitionConfig(), NDSOptimizerConfig()
    if acq == AcqType.UCB:
        return UCBAcquisitionConfig(), RAASPOptimizerConfig()
    if acq == AcqType.THOMPSON:
        return DrawAcquisitionConfig(), RAASPOptimizerConfig()
    raise ValueError("MARS designers support pareto, ucb, and thompson acquisitions")


def _init_mars_optimizer(designer: TurboENNDesigner, data, num_arms) -> None:
    designer._num_arms = num_arms
    num_init = _resolved_num_init(designer, num_arms)
    bounds = _policy_bounds(designer._policy.num_params())
    num_metrics = designer._resolve_num_metrics(data)
    config = designer._make_config(num_init, num_metrics)
    designer._turbo = create_mars_optimizer(bounds=bounds, config=config, rng=designer._rng)
    _logger.debug("Optimizer type: %s (MARS Python backend)", type(designer._turbo).__name__)


def _tell_bmars_new_data(designer: TurboBayesianMARSDesigner, new_data) -> None:
    if designer._tr_type != "morbo":
        da = __import__("optimizer.designer_asserts", fromlist=["_"])
        da.assert_scalar_rreturn(new_data)
    x_list = [d.policy.get_params() for d in new_data]
    if not x_list:
        return
    y_list = [d.trajectory.rreturn for d in new_data]
    y_se_list = [d.trajectory.rreturn_se for d in new_data]
    pass_y_var = _resolve_bmars_y_var_mode(designer, y_se_list)
    y_est = _tell_bmars_arrays(designer, x_list, y_list, y_se_list if pass_y_var else None)
    assert y_est.shape[0] == len(new_data)
    if y_est.shape[1] == 1:
        designer._update_best_estimate(new_data, y_est[:, 0])


def _resolve_bmars_y_var_mode(
    designer: TurboBayesianMARSDesigner,
    y_se_list: list[Any],
) -> bool:
    has_any = any(se is not None for se in y_se_list)
    has_all = all(se is not None for se in y_se_list)
    if has_any and not has_all:
        raise ValueError("TurboBayesianMARSDesigner requires y_var on every item in a batch or none.")
    if designer._bmars_pass_y_var is None:
        designer._bmars_pass_y_var = has_all
    if bool(designer._bmars_pass_y_var) != has_all:
        raise ValueError("TurboBayesianMARSDesigner y_var availability changed across tell calls.")
    return has_all


def _tell_bmars_arrays(
    designer: TurboBayesianMARSDesigner,
    x_list: list[np.ndarray],
    y_list: list[Any],
    y_se_list: list[Any] | None,
) -> np.ndarray:
    x = np.array(x_list)
    y_obs = np.array(y_list)
    y_obs = y_obs[:, None] if y_obs.ndim == 1 else y_obs
    if y_se_list is None:
        return designer._turbo.tell(x, y_obs)
    return designer._turbo.tell(x, y_obs, y_var=np.asarray(y_se_list) ** 2)


def _resolved_num_init(designer: TurboENNDesigner, num_arms: int) -> int:
    if designer._num_init is None:
        return int(num_arms)
    return max(int(num_arms), int(num_arms) * int(int(designer._num_init) / int(num_arms) + 0.5))


def _policy_bounds(num_dim: int) -> np.ndarray:
    from common import all_bounds as ab

    return np.array([[ab.x_low, ab.x_high]] * int(num_dim))


def _check_scalar_metrics(num_metrics: int | None, designer_name: str) -> None:
    if num_metrics not in (None, 1):
        raise ValueError(f"{designer_name} supports scalar objectives only")
