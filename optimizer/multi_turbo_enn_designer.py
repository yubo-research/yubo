from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from optimizer.multi_turbo_enn_utils import (
    call_multi_designer,
    load_multi_state,
)
from optimizer.turbo_enn_designer_ext import TurboENNDesigner


@dataclass(frozen=True)
class MultiTurboHarnessConfig:
    num_regions: int = 2
    strategy: str = "independent"
    arm_mode: str = "split"
    pool_multiplier: int = 2


@dataclass(frozen=True)
class TurboENNRegionConfig:
    turbo_mode: str
    num_init: int | None = None
    k: int | None = None
    num_keep: int | None = None
    num_fit_samples: int | None = None
    num_fit_candidates: int | None = None
    acq_type: str = "pareto"
    tr_type: str | None = None
    tr_geometry: str | None = None
    metric_sampler: str | None = None
    metric_rank: int | None = None
    pc_rotation_mode: str | None = None
    pc_rank: int | None = None
    tr_length_fixed: float | None = None
    update_option: str = "option_a"
    p_raasp: float = 0.2
    radial_mode: str = "ball_uniform"
    shape_period: int = 5
    shape_ema: float = 0.2
    rho_bad: float = 0.25
    rho_good: float = 0.75
    gamma_down: float = 0.5
    gamma_up: float = 2.0
    boundary_tol: float = 0.1
    use_y_var: bool = False
    num_candidates: int | None = None
    candidate_rv: str | None = None
    num_metrics: int | None = None


@dataclass(frozen=True)
class MultiTurboENNConfig:
    harness: MultiTurboHarnessConfig
    region: TurboENNRegionConfig


@dataclass
class MultiTurboRuntimeState:
    region_data: list[list] = field(default_factory=list)
    shared_prefix_len: int = 0
    region_assignments: list[int] = field(default_factory=list)
    last_region_indices: list[int] | None = None
    num_told_global: int = 0
    allocated_num_arms: int | None = None
    proposal_per_region: int | None = None


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
        harness = config.harness
        region = config.region

        self._turbo_mode = region.turbo_mode
        self._num_regions = int(harness.num_regions)
        if self._num_regions <= 0:
            raise ValueError(f"num_regions must be >= 1, got {harness.num_regions}")

        strategy = harness.strategy
        arm_mode = harness.arm_mode
        if strategy not in ("independent", "shared_data"):
            raise ValueError(f"Invalid strategy: {strategy}")
        if arm_mode not in ("split", "per_region", "allocated"):
            raise ValueError(f"Invalid arm_mode: {arm_mode}")
        self._strategy = strategy
        self._arm_mode = arm_mode

        self._pool_multiplier = int(harness.pool_multiplier)
        if self._pool_multiplier <= 0:
            raise ValueError(f"pool_multiplier must be > 0, got {harness.pool_multiplier}")

        self._rng = rng if rng is not None else np.random.default_rng(np.random.randint(2**31))
        self._region_rngs: list[np.random.Generator] = []
        self._designers: list[TurboENNDesigner] = []
        self._state = MultiTurboRuntimeState()

    @property
    def _acq_type(self) -> str:
        return self._config.region.acq_type

    @property
    def _tr_type(self) -> str | None:
        return self._config.region.tr_type

    def _region_designer_kwargs(self) -> dict:
        return asdict(self._config.region)

    def _init_regions(self, data, num_arms: int) -> None:
        _ = num_arms
        if self._designers:
            return
        self._state.region_data = [[] for _ in range(self._num_regions)]
        self._state.shared_prefix_len = len(data) if data else 0
        self._region_rngs = []
        self._designers = []
        designer_kwargs = self._region_designer_kwargs()
        for _ in range(self._num_regions):
            seed = int(self._rng.integers(2**31 - 1))
            child_rng = np.random.default_rng(seed)
            self._designers.append(
                TurboENNDesigner(
                    self._policy,
                    rng=child_rng,
                    **designer_kwargs,
                )
            )
            self._region_rngs.append(np.random.default_rng(int(self._rng.integers(2**31 - 1))))

        if data:
            for region_data in self._state.region_data:
                region_data.extend(data)
            self._state.num_told_global = len(data)

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
        state = self._state
        region_states = []
        for idx, designer in enumerate(self._designers):
            region_data = state.region_data[idx] if idx < len(state.region_data) else []
            region_states.append(designer.state_dict(data=region_data))
        return {
            "num_regions": self._num_regions,
            "strategy": self._strategy,
            "arm_mode": self._arm_mode,
            "pool_multiplier": self._pool_multiplier,
            "rng_state": self._rng.bit_generator.state,
            "region_rng_states": [rng.bit_generator.state for rng in self._region_rngs],
            "shared_prefix_len": int(state.shared_prefix_len),
            "num_told_global": int(state.num_told_global),
            "region_assignments": list(state.region_assignments),
            "last_region_indices": None if state.last_region_indices is None else list(state.last_region_indices),
            "allocated_num_arms": state.allocated_num_arms,
            "proposal_per_region": state.proposal_per_region,
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

    def get_algo_metrics(self) -> dict[str, float]:
        """Return optimizer-specific metrics for console display."""
        out: dict[str, float] = {}
        state = getattr(self, "_state", None)
        if state is None:
            return out
        if getattr(state, "proposal_per_region", None) is not None:
            out["region_alloc"] = float(state.proposal_per_region)
        if getattr(state, "last_region_indices", None):
            idx = state.last_region_indices
            if idx:
                out["region_idx"] = float(idx[0])
        dt_fit_total = 0.0
        dt_sel_total = 0.0
        for designer in getattr(self, "_designers", []):
            turbo = getattr(designer, "_turbo", None)
            if turbo is None or not hasattr(turbo, "telemetry"):
                continue
            tel = turbo.telemetry()
            dt_fit_total += float(getattr(tel, "dt_fit", 0.0))
            dt_sel_total += float(getattr(tel, "dt_sel", 0.0))
        if dt_fit_total > 0 or dt_sel_total > 0:
            out["fit_dt"] = dt_fit_total
            out["select_dt"] = dt_sel_total
        tr_len = None
        for designer in getattr(self, "_designers", []):
            turbo = getattr(designer, "_turbo", None)
            if turbo is not None and hasattr(turbo, "tr_length"):
                try:
                    tr_len = float(turbo.tr_length)
                    break
                except (TypeError, ValueError):
                    pass
        if tr_len is not None:
            out["tr_length"] = tr_len
        return out
