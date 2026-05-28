from __future__ import annotations

import math
from typing import Optional

from enn.turbo.config.trust_region import TurboTRConfig
from enn.turbo.python_fallback.turbo_trust_region import TurboTrustRegion

from optimizer.turbo_enn_designer import TurboENNDesigner


class _SparseOptimizerWithTrustRegion:
    """Expose sparse trust-region state while delegating ENN work to ennbo's Rust optimizer."""

    __slots__ = ("_inner", "_tr_state")

    def __init__(self, inner, tr_state: SparseEvidenceTrustRegion) -> None:
        self._inner = inner
        self._tr_state = tr_state

    def ask(self, num_arms: int):
        return self._inner.ask(num_arms)

    def tell(self, x, y, y_var=None):
        if y_var is None:
            return self._inner.tell(x, y)
        return self._inner.tell(x, y, y_var=y_var)

    def telemetry(self):
        return self._inner.telemetry()

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


class SparseEvidenceTrustRegion(TurboTrustRegion):
    """TuRBO trust region with a sparse-proposal failure clock.

    Candidate generation still uses FAST RAASP.  The only behavioral change is
    the failure tolerance: it scales with the expected active RAASP support
    rather than the ambient dimension.
    """

    def __init__(
        self,
        config: TurboTRConfig,
        num_dim: int,
        *,
        num_pert: int = 20,
        clock_scale: float = 3.0,
        min_failures: float = 4.0,
        incumbent_selector=None,
    ) -> None:
        if int(num_dim) <= 0:
            raise ValueError(f"num_dim must be positive, got {num_dim}")
        if int(num_pert) <= 0:
            raise ValueError(f"num_pert must be positive, got {num_pert}")
        if float(clock_scale) <= 0.0:
            raise ValueError(f"clock_scale must be positive, got {clock_scale}")
        if float(min_failures) <= 0.0:
            raise ValueError(f"min_failures must be positive, got {min_failures}")
        self.num_pert = int(num_pert)
        self.clock_scale = float(clock_scale)
        self.min_failures = float(min_failures)
        super().__init__(
            config=config,
            num_dim=int(num_dim),
            incumbent_selector=incumbent_selector,
        )

    @property
    def expected_support(self) -> float:
        """Expected support of max(1, Binomial(D, min(num_pert / D, 1)))."""
        if self.num_dim <= self.num_pert:
            return float(self.num_dim)
        p = float(self.num_pert) / float(self.num_dim)
        return float(self.num_dim) * p + math.exp(float(self.num_dim) * math.log1p(-p))

    def _ensure_initialized(self, num_arms: int) -> None:
        num_arms = int(num_arms)
        if num_arms <= 0:
            raise ValueError(num_arms)
        if self._num_arms is None:
            tolerance = max(
                1.0,
                self.min_failures / float(num_arms),
                self.clock_scale * self.expected_support / float(num_arms),
            )
            self._num_arms = num_arms
            self._failure_tolerance = int(math.ceil(tolerance))
        elif num_arms != self._num_arms:
            raise ValueError(f"num_arms changed from {self._num_arms} to {num_arms}; must be consistent across ask() calls")
        assert self._failure_tolerance is not None


class SparseENNDesigner(TurboENNDesigner):
    """ENN local optimizer with sparse-proposal trust-region clock."""

    def __init__(
        self,
        policy,
        *,
        clock_scale: float = 3.0,
        num_pert: int = 20,
        min_failures: float = 4.0,
        num_init: Optional[int] = None,
        k: Optional[int] = 10,
        num_keep: Optional[int] = None,
        num_fit_samples: Optional[int] = None,
        num_fit_candidates: Optional[int] = None,
        acq_type: str = "pareto",
        use_y_var: bool = False,
        num_candidates: Optional[int] = None,
        candidate_rv: Optional[str] = None,
    ) -> None:
        if int(num_pert) <= 0:
            raise ValueError(f"num_pert must be positive, got {num_pert}")
        if float(clock_scale) <= 0.0:
            raise ValueError(f"clock_scale must be positive, got {clock_scale}")
        if float(min_failures) <= 0.0:
            raise ValueError(f"min_failures must be positive, got {min_failures}")
        self._clock_scale = float(clock_scale)
        self._sparse_num_pert = int(num_pert)
        self._min_failures = float(min_failures)
        super().__init__(
            policy,
            turbo_mode="turbo-enn",
            num_init=num_init,
            k=k,
            num_keep=num_keep,
            num_fit_samples=num_fit_samples,
            num_fit_candidates=num_fit_candidates,
            acq_type=acq_type,
            tr_type="turbo",
            use_y_var=use_y_var,
            num_candidates=num_candidates,
            candidate_rv=candidate_rv,
        )

    def _init_optimizer(self, data, num_arms):
        super()._init_optimizer(data, num_arms)
        tr_state = SparseEvidenceTrustRegion(
            TurboTRConfig(),
            self._policy.num_params(),
            num_pert=self._sparse_num_pert,
            clock_scale=self._clock_scale,
            min_failures=self._min_failures,
        )
        tr_state.validate_request(num_arms)
        self._turbo = _SparseOptimizerWithTrustRegion(self._turbo, tr_state)
