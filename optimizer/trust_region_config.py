from __future__ import annotations

from typing import Any, Literal

from attrs import define, field
from attrs import validators as v
from enn.turbo.components.incumbent_selector import ScalarIncumbentSelector
from enn.turbo.config.candidate_rv import CandidateRV
from enn.turbo.config.tr_length_config import TRLengthConfig
from enn.turbo.turbo_trust_region import TurboTrustRegion

from optimizer.box_trust_region import FixedLengthTurboTrustRegion
from optimizer.ellipsoidal_trust_region import ENNTrueEllipsoidalTrustRegion
from optimizer.metric_trust_region import (
    ENNMetricShapedTrustRegion,
    MetricShapedTrustRegion,
)
from optimizer.pc_rotation import PCRotationMode
from optimizer.trust_region_jax import is_available as _jax_available
from optimizer.trust_region_jax import set_backend as _set_backend
from optimizer.trust_region_utils import (
    RadialMode,
    SamplerKind,
    UpdateMode,
    _ray_scale_to_unit_box,
)

GeometryKind = Literal[
    "box",
    "enn_metr",
    "grad_metr",
    "enn_ellip",
    "grad_ellip",
    "enn_metric_shaped",
    "enn_grad_metric_shaped",
    "enn_true_ellipsoid",
    "enn_grad_true_ellipsoid",
]

_GEOMETRY_ALIASES = {
    "box": "box",
    "enn_metr": "enn_metr",
    "grad_metr": "grad_metr",
    "enn_ellip": "enn_ellip",
    "grad_ellip": "grad_ellip",
    "enn_metric_shaped": "enn_metr",
    "enn_grad_metric_shaped": "grad_metr",
    "enn_true_ellipsoid": "enn_ellip",
    "enn_grad_true_ellipsoid": "grad_ellip",
}
_GEOMETRIES = frozenset(("box", "enn_metr", "grad_metr", "enn_ellip", "grad_ellip"))
_RADIAL = frozenset(("ball_uniform", "boundary"))


def normalize_geometry_name(geometry: str) -> str:
    key = str(geometry).strip()
    return _GEOMETRY_ALIASES.get(key, key)


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
    use_jax_tr: bool = False
    tr_backend: str | None = None

    def __attrs_post_init__(self) -> None:
        normalized_geometry = normalize_geometry_name(self.geometry)
        object.__setattr__(self, "geometry", normalized_geometry)
        if self.geometry not in _GEOMETRIES:
            raise ValueError(f"Unknown geometry={self.geometry!r}")
        if self.geometry == "box":
            if self.metric_sampler is not None or self.metric_rank is not None:
                raise ValueError("metric_* options require non-box geometry")
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
        if self.tr_backend:
            _set_backend(self.tr_backend)
        use_jax = (bool(self.use_jax_tr) or bool(self.tr_backend)) and _jax_available()
        if self.geometry in ("enn_ellip", "grad_ellip"):
            return ENNTrueEllipsoidalTrustRegion(
                config=self,
                num_dim=num_dim,
                incumbent_selector=selector,
                candidate_rv=candidate_rv,
                metric_sampler=sampler,
                metric_rank=self.metric_rank,
                use_jax=use_jax,
            )
        if self.geometry in ("enn_metr", "grad_metr"):
            return ENNMetricShapedTrustRegion(
                config=self,
                num_dim=num_dim,
                incumbent_selector=selector,
                candidate_rv=candidate_rv,
                metric_sampler=sampler,
                metric_rank=self.metric_rank,
                use_jax=use_jax,
            )
        raise ValueError(f"Unknown geometry={self.geometry!r}")


__all__ = [
    "MetricShapedTRConfig",
    "MetricShapedTrustRegion",
    "ENNMetricShapedTrustRegion",
    "ENNTrueEllipsoidalTrustRegion",
    "normalize_geometry_name",
    "_ray_scale_to_unit_box",
]
