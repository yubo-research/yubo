from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optimizer.designer_asserts import assert_scalar_rreturn
from optimizer.multi_turbo_enn_allocation import (
    allocated_proposal_plan,
    fixed_region_counts,
    propose_batch,
    select_top_k,
)
from optimizer.multi_turbo_enn_scoring import (
    score_multi_candidates as score_multi_candidates_pure,
)
from optimizer.multi_turbo_enn_state import (
    RegionToleranceTarget,
    SelectionCommit,
    _tell_new_data_if_any,
    commit_selection,
    load_multi_state,
)

if TYPE_CHECKING:
    from optimizer.multi_turbo_enn_designer import MultiTurboENNDesigner


__all__ = [
    "_call_fixed_arms",
    "_tell_new_data_if_any",
    "call_multi_designer",
    "load_multi_state",
]


def _call_allocated(
    designer: MultiTurboENNDesigner,
    *,
    num_arms: int,
) -> list:
    plan = allocated_proposal_plan(
        num_arms=num_arms,
        num_regions=designer._num_regions,
        pool_multiplier=designer._pool_multiplier,
        allocated_num_arms=designer._state.allocated_num_arms,
        proposal_per_region=designer._state.proposal_per_region,
    )
    batch = propose_batch(
        designer._designers,
        designer._state.region_data,
        per_region=plan.per_region,
    )
    if not batch.policies:
        return []
    if len(batch.policies) <= num_arms:
        commit_selection(
            designer,
            SelectionCommit(
                pending_region_indices=batch.region_indices,
                allocated_num_arms=plan.allocated_num_arms,
                proposal_per_region=plan.proposal_per_region,
            ),
        )
        return batch.policies
    x_all = np.array([policy.get_params() for policy in batch.policies], dtype=float)
    scores = score_multi_candidates_pure(
        x_all,
        batch.region_indices,
        child_designers=designer._designers,
        region_data_lens=[len(data) for data in designer._state.region_data],
        region_rngs=designer._region_rngs,
        acq_type=designer._acq_type,
        rng=designer._rng,
    )
    chosen = select_top_k(scores, k=num_arms)
    chosen_regions = [batch.region_indices[int(i)] for i in chosen]
    tolerance_targets = _extract_tolerance_targets(designer._designers)
    commit_selection(
        designer,
        SelectionCommit(
            pending_region_indices=chosen_regions,
            allocated_num_arms=plan.allocated_num_arms,
            proposal_per_region=plan.proposal_per_region,
            update_failure_tolerances=True,
            tolerance_targets=tolerance_targets,
        ),
    )
    return [batch.policies[int(i)] for i in chosen]


def _call_fixed_arms(
    designer: MultiTurboENNDesigner,
    *,
    num_arms: int,
) -> list:
    per_region_fn = getattr(designer, "_per_region_counts", None)
    per_region = fixed_region_counts(
        num_arms=num_arms,
        num_regions=getattr(designer, "_num_regions", len(designer._designers)),
        arm_mode=getattr(designer, "_arm_mode", "split"),
        per_region_fn=per_region_fn if callable(per_region_fn) else None,
    )
    batch = propose_batch(designer._designers, designer._state.region_data, per_region=per_region)
    commit_selection(designer, SelectionCommit(pending_region_indices=batch.region_indices))
    return batch.policies


def _extract_tolerance_targets(child_designers: list) -> list[RegionToleranceTarget]:
    targets: list[RegionToleranceTarget] = []
    for child in child_designers:
        turbo = getattr(child, "_turbo", None)
        tr_state = getattr(turbo, "_tr_state", None) if turbo is not None else None
        if tr_state is None or not hasattr(tr_state, "failure_tolerance"):
            targets.append(RegionToleranceTarget(num_dim=0))
            continue
        num_dim = int(getattr(tr_state, "num_dim", 0))
        if num_dim <= 0:
            targets.append(RegionToleranceTarget(num_dim=0))
            continue

        def _setter(new_tol: int, *, _tr_state=tr_state) -> None:
            _tr_state.failure_tolerance = int(new_tol)

        targets.append(
            RegionToleranceTarget(
                num_dim=num_dim,
                set_failure_tolerance=_setter,
            )
        )
    return targets


def call_multi_designer(
    designer: MultiTurboENNDesigner,
    data: list,
    *,
    num_arms: int,
    telemetry=None,
) -> list:
    if designer._tr_type != "morbo":
        assert_scalar_rreturn(data)
    if num_arms <= 0:
        raise ValueError(num_arms)

    designer._init_regions(data, num_arms)
    _tell_new_data_if_any(designer, data)
    if designer._arm_mode == "allocated":
        policies = _call_allocated(designer, num_arms=num_arms)
    else:
        policies = _call_fixed_arms(designer, num_arms=num_arms)

    if telemetry is not None:
        designer._set_telemetry(telemetry)
    return policies
