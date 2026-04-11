from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from enn.turbo.config.candidate_rv import CandidateRV
from enn.turbo.turbo_trust_region import TurboTrustRegion

from optimizer.submodule_perturbator import leaf_module_param_blocks
from optimizer.trust_region_utils import _generate_block_raasp_candidates


class FixedLengthTurboTrustRegion(TurboTrustRegion):
    def _apply_fixed_length(self) -> None:
        fixed_length = getattr(self.config, "fixed_length", None)
        if fixed_length is not None:
            self.length = float(fixed_length)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._apply_fixed_length()

    def update(self, y_obs: Any, y_incumbent: Any) -> None:
        super().update(y_obs, y_incumbent)
        self._apply_fixed_length()

    def restart(self, rng: Any | None = None) -> None:
        super().restart(rng=rng)
        self._apply_fixed_length()


@dataclass
class ModuleAwareTrustRegion(FixedLengthTurboTrustRegion):
    block_slices: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    block_prob: float = 0.5
    uses_custom_candidate_gen: bool = field(default=True, init=False)

    def generate_candidates(
        self,
        x_center,
        lengthscales,
        num_candidates,
        *,
        rng: Any,
        candidate_rv: CandidateRV | None = None,
        sobol_engine: Any | None = None,
        raasp_driver: Any | None = None,
        num_pert: int = 20,
    ):
        _ = raasp_driver, num_pert
        rv = CandidateRV.UNIFORM if candidate_rv is None else candidate_rv
        lb, ub = self.compute_bounds_1d(x_center, lengthscales)
        return _generate_block_raasp_candidates(
            x_center,
            lb,
            ub,
            int(num_candidates),
            rng=rng,
            candidate_rv=rv,
            sobol_engine=sobol_engine,
            block_slices=self.block_slices,
            block_prob=float(self.block_prob),
        )


def maybe_enable_module_masks(
    optimizer: Any,
    policy: Any,
    *,
    enabled: bool = False,
    min_num_params: int = 10000,
    block_prob: float = 0.5,
) -> bool:
    if not enabled:
        return False

    try:
        import torch.nn as nn
    except ImportError:
        return False

    if not isinstance(policy, nn.Module):
        return False
    num_params = getattr(policy, "num_params", None)
    if not callable(num_params) or int(num_params()) < int(min_num_params):
        return False

    tr_state = getattr(optimizer, "_tr_state", None)
    if tr_state is None:
        return False
    if not isinstance(tr_state, TurboTrustRegion):
        return False

    if getattr(tr_state, "uses_custom_candidate_gen", False):
        return bool(getattr(tr_state, "module_block_slices", ()))

    block_slices = leaf_module_param_blocks(policy)
    if len(block_slices) < 2:
        return False

    replacement = ModuleAwareTrustRegion(
        config=tr_state.config,
        num_dim=int(tr_state.num_dim),
        incumbent_selector=getattr(tr_state, "incumbent_selector", None),
        block_slices=block_slices,
        block_prob=float(block_prob),
    )
    replacement.length = float(tr_state.length)
    replacement.failure_counter = int(tr_state.failure_counter)
    replacement.success_counter = int(tr_state.success_counter)
    replacement.best_value = float(tr_state.best_value)
    replacement.prev_num_obs = int(tr_state.prev_num_obs)
    replacement._num_arms = getattr(tr_state, "_num_arms", None)
    replacement._failure_tolerance = getattr(tr_state, "_failure_tolerance", None)
    optimizer._tr_state = replacement
    return True
