from __future__ import annotations

from typing import Any, Literal

from attrs import define, field
from attrs import validators as v
from enn.turbo.components.incumbent_selector import ScalarIncumbentSelector
from enn.turbo.config.candidate_rv import CandidateRV
from enn.turbo.config.tr_length_config import TRLengthConfig
from enn.turbo.turbo_trust_region import TurboTrustRegion

from optimizer.box_trust_region import FixedLengthTurboTrustRegion
from optimizer.ellipsoidal_trust_region import (
    ENNGradientIsotropicTrustRegion,
    ENNIsotropicTrustRegion,
    ENNTrueEllipsoidalTrustRegion,
)
from optimizer.metric_trust_region import (
    ENNMetricShapedTrustRegion,
    MetricShapedTrustRegion,
)
from optimizer.pc_rotation import PCRotationMode
from optimizer.trust_region_accel import accel_name as _accel_name
from optimizer.trust_region_accel import accel_override as _accel_override
from optimizer.trust_region_utils import (
    RadialMode,
    SamplerKind,
    UpdateMode,
    _ray_scale_to_unit_box,
)

GeometryKind = Literal[
    "box",
    "enn_iso",
    "grad_iso",
    "enn_metr",
    "grad_metr",
    "enn_ellip",
    "grad_ellip",
]
_GEOMETRIES = frozenset(("box", "enn_iso", "grad_iso", "enn_metr", "grad_metr", "enn_ellip", "grad_ellip"))
_RADIAL = frozenset(("ball_uniform", "boundary"))


def normalize_geometry_name(geometry: str) -> str:
    return str(geometry).strip()


@define(frozen=True)
class MetricShapedTRConfig:
    length: TRLengthConfig = field(factory=TRLengthConfig)
    fixed_length: float | None = None
    noise_aware: bool = False
    geometry: GeometryKind = "box"
    metric_sampler: SamplerKind | None = None
    metric_rank: int | None = field(default=None, validator=v.optional(v.gt(0)))
    pc_rotation_mode: PCRotationMode | None = field(
        default=None,
        validator=v.optional(v.in_(["full", "low_rank"])),
    )
    pc_rank: int | None = field(default=None, validator=v.optional(v.gt(0)))
    update_option: UpdateMode = "option_a"
    p_raasp: float = field(default=0.2, validator=v.and_(v.gt(0), v.le(1)))
    radial_mode: RadialMode = field(default="ball_uniform", validator=v.in_(_RADIAL))
    shape_period: int = field(default=5, validator=v.gt(0))
    shape_ema: float = field(default=0.2, validator=v.and_(v.gt(0), v.le(1)))
    shape_jitter: float = field(default=1e-6, validator=v.gt(0))
    shape_kappa_max: float = field(default=1e4, validator=v.ge(1))
    rho_bad: float = 0.25
    rho_good: float = 0.75
    gamma_down: float = field(default=0.5, validator=v.and_(v.gt(0), v.lt(1)))
    gamma_up: float = field(default=2.0, validator=v.gt(1))
    boundary_tol: float = field(default=0.1, validator=v.and_(v.ge(0), v.le(1)))
    use_accel: bool = False
    accel: str | None = None

    def __attrs_post_init__(self) -> None:
        normalized_geometry = normalize_geometry_name(self.geometry)
        object.__setattr__(self, "geometry", normalized_geometry)
        if self.geometry not in _GEOMETRIES:
            raise ValueError(f"Unknown geometry={self.geometry!r}")
        if self.geometry == "box":
            if self.metric_sampler is not None or self.metric_rank is not None:
                raise ValueError("metric_* options require non-box geometry")
        if self.geometry in {"enn_iso", "grad_iso"}:
            if self.metric_sampler not in (None, "low_rank"):
                raise ValueError(f"{self.geometry} only supports sampler='low_rank' or the default sampler.")
            if self.metric_rank is not None:
                raise ValueError(f"{self.geometry} does not use metric_rank.")
        if self.geometry in {"enn_metr", "grad_metr"}:
            if self.radial_mode != "ball_uniform":
                raise ValueError(f"{self.geometry} only supports radial_mode='ball_uniform'; got {self.radial_mode!r}")
        if self.fixed_length is not None and float(self.fixed_length) <= 0:
            raise ValueError(f"fixed_length must be > 0, got {self.fixed_length}")

    @property
    def length_init(self) -> float:
        return self.length.length_init

    @property
    def length_min(self) -> float:
        return self.length.length_min

    @property
    def length_max(self) -> float:
        return self.length.length_max

    def _resolved_sampler(self) -> SamplerKind:
        return "full" if self.metric_sampler is None else self.metric_sampler

    def build(
        self,
        *,
        num_dim: int,
        rng: Any,
        candidate_rv: CandidateRV | None = None,
    ) -> TurboTrustRegion:
        _ = rng
        selector = ScalarIncumbentSelector(noise_aware=self.noise_aware)
        if self.geometry == "box":
            return FixedLengthTurboTrustRegion(
                config=self,
                num_dim=num_dim,
                incumbent_selector=selector,
            )
        sampler = self._resolved_sampler()
        candidate_rv = CandidateRV.SOBOL if candidate_rv is None else candidate_rv
        accel_requested = bool(self.use_accel) or bool(self.accel)
        with _accel_override(self.accel):
            use_accel = accel_requested and _accel_name() != "none"
        if self.geometry == "enn_iso":
            # The identity-metric control is fastest in the empty low-rank
            # representation: it preserves the isotropic ball geometry while
            # avoiding dense factor/solve work on the proposal path.
            sampler = "low_rank"
            return ENNIsotropicTrustRegion(
                config=self,
                num_dim=num_dim,
                incumbent_selector=selector,
                candidate_rv=candidate_rv,
                metric_sampler=sampler,
                metric_rank=self.metric_rank,
                use_accel=use_accel,
            )
        if self.geometry == "grad_iso":
            sampler = "low_rank"
            return ENNGradientIsotropicTrustRegion(
                config=self,
                num_dim=num_dim,
                incumbent_selector=selector,
                candidate_rv=candidate_rv,
                metric_sampler=sampler,
                metric_rank=self.metric_rank,
                use_accel=use_accel,
            )
        if self.geometry in ("enn_ellip", "grad_ellip"):
            return ENNTrueEllipsoidalTrustRegion(
                config=self,
                num_dim=num_dim,
                incumbent_selector=selector,
                candidate_rv=candidate_rv,
                metric_sampler=sampler,
                metric_rank=self.metric_rank,
                use_accel=use_accel,
            )
        if self.geometry in ("enn_metr", "grad_metr"):
            return ENNMetricShapedTrustRegion(
                config=self,
                num_dim=num_dim,
                incumbent_selector=selector,
                candidate_rv=candidate_rv,
                metric_sampler=sampler,
                metric_rank=self.metric_rank,
                use_accel=use_accel,
            )
        raise ValueError(f"Unknown geometry={self.geometry!r}")


__all__ = [
    "MetricShapedTRConfig",
    "MetricShapedTrustRegion",
    "ENNIsotropicTrustRegion",
    "ENNGradientIsotropicTrustRegion",
    "ENNMetricShapedTrustRegion",
    "ENNTrueEllipsoidalTrustRegion",
    "normalize_geometry_name",
    "_ray_scale_to_unit_box",
]
