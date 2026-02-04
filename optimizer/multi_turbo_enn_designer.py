from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimizer.multi_turbo_enn_utils import (
    call_multi_designer,
    load_multi_state,
    score_multi_candidates,
)
from optimizer.turbo_enn_designer import TurboENNDesigner


@dataclass(frozen=True)
class MultiTurboENNConfig:
    turbo_mode: str
    num_regions: int = 2
    strategy: str = "independent"
    arm_mode: str = "split"
    pool_multiplier: int = 2
    num_init: int | None = None
    k: int | None = None
    num_keep: int | None = None
    num_fit_samples: int | None = None
    num_fit_candidates: int | None = None
    acq_type: str = "pareto"
    tr_type: str | None = None
    tr_geometry: str | None = None
    ellipsoid_sampler: str | None = None
    ellipsoid_rank: int | None = None
    use_y_var: bool = False
    num_candidates: int | None = None
    candidate_rv: str | None = None
    num_metrics: int | None = None


class MultiTurboENNDesigner:
    def __init__(
        self,
        policy,
        *,
        config: MultiTurboENNConfig,
        rng: np.random.Generator | None = None,
    ):
        self._policy = policy
        self._config = config
        self._turbo_mode = config.turbo_mode
        self._num_regions = int(config.num_regions)
        if self._num_regions <= 0:
            raise ValueError(f"num_regions must be >= 1, got {config.num_regions}")

        strategy = config.strategy
        arm_mode = config.arm_mode
        if strategy not in ("independent", "shared_data"):
            raise ValueError(f"Invalid strategy: {strategy}")
        if arm_mode not in ("split", "per_region", "allocated"):
            raise ValueError(f"Invalid arm_mode: {arm_mode}")
        self._strategy = strategy
        self._arm_mode = arm_mode

        self._pool_multiplier = int(config.pool_multiplier)
        if self._pool_multiplier <= 0:
            raise ValueError(f"pool_multiplier must be > 0, got {config.pool_multiplier}")

        (
            self._num_init,
            self._k,
            self._num_keep,
            self._num_fit_samples,
            self._num_fit_candidates,
            self._acq_type,
            self._tr_type,
            self._tr_geometry,
            self._ellipsoid_sampler,
            self._ellipsoid_rank,
            self._use_y_var,
            self._num_candidates,
            self._candidate_rv,
            self._num_metrics,
        ) = (
            config.num_init,
            config.k,
            config.num_keep,
            config.num_fit_samples,
            config.num_fit_candidates,
            config.acq_type,
            config.tr_type,
            config.tr_geometry,
            config.ellipsoid_sampler,
            config.ellipsoid_rank,
            config.use_y_var,
            config.num_candidates,
            config.candidate_rv,
            config.num_metrics,
        )

        self._rng = rng if rng is not None else np.random.default_rng(np.random.randint(2**31))
        self._region_rngs: list[np.random.Generator] = []
        self._designers: list[TurboENNDesigner] = []
        self._region_data: list[list] = []
        self._shared_prefix_len = 0
        self._region_assignments: list[int] = []
        self._last_region_indices: list[int] | None = None
        self._num_told_global = 0
        self._allocated_num_arms: int | None = None
        self._proposal_per_region: int | None = None

    def _init_regions(self, data, num_arms: int) -> None:
        if self._designers:
            return
        self._region_data = [[] for _ in range(self._num_regions)]
        self._shared_prefix_len = len(data) if data else 0
        self._region_rngs = []
        self._designers = []
        for _ in range(self._num_regions):
            seed = int(self._rng.integers(2**31 - 1))
            rng = np.random.default_rng(seed)
            designer = TurboENNDesigner(
                self._policy,
                turbo_mode=self._turbo_mode,
                num_init=self._num_init,
                k=self._k,
                num_keep=self._num_keep,
                num_fit_samples=self._num_fit_samples,
                num_fit_candidates=self._num_fit_candidates,
                acq_type=self._acq_type,
                tr_type=self._tr_type,
                tr_geometry=self._tr_geometry,
                ellipsoid_sampler=self._ellipsoid_sampler,
                ellipsoid_rank=self._ellipsoid_rank,
                use_y_var=self._use_y_var,
                num_candidates=self._num_candidates,
                candidate_rv=self._candidate_rv,
                num_metrics=self._num_metrics,
                rng=rng,
            )
            self._region_rngs.append(np.random.default_rng(int(self._rng.integers(2**31 - 1))))
            self._designers.append(designer)

        if data:
            self._broadcast_new_data(data)
            self._num_told_global = len(data)

    def _broadcast_new_data(self, new_data) -> None:
        for region_data in self._region_data:
            region_data.extend(new_data)

    def _assign_new_data(self, new_data) -> None:
        if self._last_region_indices is None:
            raise RuntimeError("Missing region assignments for new data")
        if len(new_data) > len(self._last_region_indices):
            raise RuntimeError("More new data than previous proposals")
        for datum, region_idx in zip(new_data, self._last_region_indices, strict=True):
            if region_idx < 0 or region_idx >= self._num_regions:
                raise RuntimeError(f"Invalid region index {region_idx}")
            self._region_data[region_idx].append(datum)
            self._region_assignments.append(int(region_idx))
        self._last_region_indices = None

    def _per_region_counts(self, num_arms: int) -> list[int]:
        if self._arm_mode == "per_region":
            return [num_arms] * self._num_regions
        if num_arms < self._num_regions:
            raise ValueError(f"num_arms={num_arms} must be >= num_regions={self._num_regions} when arm_mode='split'")
        base = num_arms // self._num_regions
        remainder = num_arms % self._num_regions
        return [base + (1 if i < remainder else 0) for i in range(self._num_regions)]

    def _predict_mu_sigma(self, region_idx: int, x_region: np.ndarray):
        return self._designers[region_idx].predict_mu_sigma(x_region)

    def _scalarize(self, region_idx: int, y: np.ndarray) -> np.ndarray | None:
        scalarize = getattr(self._designers[region_idx], "scalarize", None)
        if scalarize is None or not callable(scalarize):
            return None
        return scalarize(y, clip=False)

    def _score_candidates(self, x_all: np.ndarray, region_indices: list[int]) -> np.ndarray:
        return score_multi_candidates(self, x_all, region_indices)

    def _adjust_failure_tolerance(self, region_idx: int, num_assigned: int) -> None:
        if self._arm_mode != "allocated" or num_assigned <= 0:
            return
        designer = self._designers[region_idx]
        turbo = getattr(designer, "_turbo", None)
        tr_state = getattr(turbo, "_tr_state", None) if turbo is not None else None
        if tr_state is None:
            return
        num_dim = int(getattr(tr_state, "num_dim", 0))
        if num_dim <= 0:
            return
        try:
            tr_state.failure_tolerance = int(np.ceil(max(4.0 / num_assigned, num_dim / num_assigned)))
        except Exception:
            return

    def best_datum(self):
        best = None
        best_val = None
        for designer in self._designers:
            datum = designer.best_datum()
            if datum is None:
                continue
            val = float(datum.trajectory.get_decision_rreturn())
            if best is None or best_val is None or val > best_val:
                best = datum
                best_val = val
        return best

    def state_dict(self, data=None) -> dict:
        if data is None:
            data = []
        region_states = []
        for idx, designer in enumerate(self._designers):
            region_data = self._region_data[idx] if idx < len(self._region_data) else []
            region_states.append(designer.state_dict(data=region_data))
        return {
            "num_regions": self._num_regions,
            "strategy": self._strategy,
            "arm_mode": self._arm_mode,
            "pool_multiplier": self._pool_multiplier,
            "rng_state": self._rng.bit_generator.state,
            "region_rng_states": [rng.bit_generator.state for rng in self._region_rngs],
            "shared_prefix_len": int(self._shared_prefix_len),
            "num_told_global": int(self._num_told_global),
            "region_assignments": list(self._region_assignments),
            "last_region_indices": None if self._last_region_indices is None else list(self._last_region_indices),
            "allocated_num_arms": self._allocated_num_arms,
            "proposal_per_region": self._proposal_per_region,
            "region_states": region_states,
        }

    def load_state_dict(self, state: dict, *, data=None) -> None:
        if data is None:
            data = []
        load_multi_state(self, state, list(data))

    def stop(self) -> None:
        for designer in self._designers:
            if hasattr(designer, "stop"):
                designer.stop()

    def __call__(self, data, num_arms, *, telemetry=None):
        return call_multi_designer(
            self,
            list(data),
            num_arms=int(num_arms),
            telemetry=telemetry,
        )

    def _set_telemetry(self, telemetry) -> None:
        dt_fit_total = 0.0
        dt_sel_total = 0.0
        for designer in self._designers:
            turbo = getattr(designer, "_turbo", None)
            if turbo is None or not hasattr(turbo, "telemetry"):
                continue
            tel = turbo.telemetry()
            dt_fit_total += float(getattr(tel, "dt_fit", 0.0))
            dt_sel_total += float(getattr(tel, "dt_sel", 0.0))
        telemetry.set_dt_fit(dt_fit_total)
        telemetry.set_dt_select(dt_sel_total)
