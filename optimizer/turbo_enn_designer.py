import pickle
from typing import Callable, Optional

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
)

import common.all_bounds as all_bounds
from optimizer.designer_asserts import assert_scalar_rreturn
from optimizer.ellipsoid_trust_region import EllipsoidTRConfig
from optimizer.enn_turbo_optimizer import (
    create_optimizer,
)
from optimizer.enn_turbo_optimizer import (
    predict_mu_sigma as turbo_predict_mu_sigma,
)
from optimizer.enn_turbo_optimizer import (
    scalarize as turbo_scalarize,
)


def _validate_tr_options(
    *,
    geometry: str,
    sampler: str | None,
    rank: int | None,
    fixed_length: float | None,
) -> None:
    allowed_geometry = {
        "box",
        "enn_ellipsoid",
        "enn_grad_ellipsoid",
    }
    allowed_sampler = {"full", "low_rank"}
    if geometry not in allowed_geometry:
        raise ValueError(f"Invalid tr_geometry: {geometry}")
    if sampler is not None and sampler not in allowed_sampler:
        raise ValueError(f"Invalid ellipsoid_sampler: {sampler}")
    if geometry == "box":
        if sampler is not None or rank is not None:
            raise ValueError("ellipsoid_* options require non-box geometry")
        if fixed_length is not None:
            raise ValueError("tr_length_fixed requires ellipsoid geometry")
    if geometry == "enn_grad_ellipsoid":
        if sampler is not None and sampler != "low_rank":
            raise ValueError("enn_grad_ellipsoid requires ellipsoid_sampler='low_rank'")
    if rank is not None and int(rank) <= 0:
        raise ValueError(f"ellipsoid_rank must be > 0, got {rank}")
    if fixed_length is not None and float(fixed_length) <= 0:
        raise ValueError(f"tr_length_fixed must be > 0, got {fixed_length}")


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
        return EllipsoidTRConfig(
            geometry=designer._tr_geometry,
            ellipsoid_sampler=designer._ellipsoid_sampler,
            ellipsoid_rank=designer._ellipsoid_rank,
            fixed_length=designer._tr_length_fixed,
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
        ellipsoid_sampler: Optional[str] = None,
        ellipsoid_rank: Optional[int] = None,
        tr_length_fixed: Optional[float] = None,
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
        self._ellipsoid_sampler = ellipsoid_sampler
        self._ellipsoid_rank = ellipsoid_rank
        self._tr_length_fixed = tr_length_fixed
        self._use_y_var = use_y_var
        self._num_candidates = num_candidates
        self._candidate_rv = candidate_rv
        self._num_metrics = num_metrics

        if self._num_fit_samples is not None and int(self._num_fit_samples) <= 0:
            raise ValueError(f"num_fit_samples must be > 0, got {self._num_fit_samples}")
        if self._num_fit_candidates is not None and int(self._num_fit_candidates) <= 0:
            raise ValueError(f"num_fit_candidates must be > 0, got {self._num_fit_candidates}")

        self._turbo = None
        self._num_arms = None
        self._rng = rng if rng is not None else np.random.default_rng(np.random.randint(2**31))
        self._num_told = 0
        self._datum_best = None
        self._y_est_best = None

        _validate_tr_options(
            geometry=self._tr_geometry,
            sampler=self._ellipsoid_sampler,
            rank=self._ellipsoid_rank,
            fixed_length=self._tr_length_fixed,
        )

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
        bounds = np.array([[all_bounds.x_low, all_bounds.x_high]] * num_dim)
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
        return policy
