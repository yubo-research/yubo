from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from attrs import define

if TYPE_CHECKING:
    from optimizer.multi_turbo_enn_designer import MultiTurboENNDesigner


@define(frozen=True)
class SelectionCommit:
    pending_region_indices: list[int]
    allocated_num_arms: int | None = None
    proposal_per_region: int | None = None
    update_failure_tolerances: bool = False
    tolerance_targets: list["RegionToleranceTarget"] | None = None


@define(frozen=True)
class RegionToleranceTarget:
    num_dim: int
    set_failure_tolerance: Callable[[int], None] | None = None


def _append_pending_indices(designer: MultiTurboENNDesigner, region_indices: list[int]) -> None:
    state = designer._state
    if state.last_region_indices is None:
        state.last_region_indices = [int(i) for i in region_indices]
    else:
        state.last_region_indices.extend(int(i) for i in region_indices)


def commit_selection(designer: MultiTurboENNDesigner, commit: SelectionCommit) -> None:
    state = designer._state
    if commit.allocated_num_arms is not None:
        state.allocated_num_arms = int(commit.allocated_num_arms)
    if commit.proposal_per_region is not None:
        state.proposal_per_region = int(commit.proposal_per_region)
    _append_pending_indices(designer, commit.pending_region_indices)
    if commit.update_failure_tolerances:
        _update_allocated_failure_tolerances(
            commit.pending_region_indices,
            tolerance_targets=commit.tolerance_targets,
        )


def _update_allocated_failure_tolerances(
    chosen_regions: list[int],
    *,
    tolerance_targets: list[RegionToleranceTarget] | None,
) -> None:
    if not tolerance_targets:
        return
    counts = np.bincount(np.asarray(chosen_regions, dtype=int), minlength=len(tolerance_targets))
    for region_idx, num_assigned in enumerate(counts.tolist()):
        if num_assigned <= 0:
            continue
        target = tolerance_targets[region_idx]
        num_dim = int(target.num_dim)
        if num_dim <= 0:
            continue
        setter = target.set_failure_tolerance
        if setter is None:
            continue
        new_tol = int(np.ceil(max(4.0 / num_assigned, num_dim / num_assigned)))
        try:
            setter(new_tol)
        except AttributeError:
            continue


def _restore_rng_states(designer: MultiTurboENNDesigner, state: dict) -> None:
    rng_state = state.get("rng_state")
    if rng_state is not None:
        designer._rng.bit_generator.state = rng_state

    region_rng_states = state.get("region_rng_states")
    if region_rng_states is None:
        return
    if len(region_rng_states) != len(designer._region_rngs):
        raise ValueError("region_rng_states has wrong length")
    for rng, region_state in zip(designer._region_rngs, region_rng_states, strict=True):
        rng.bit_generator.state = region_state


def _load_runtime_fields(runtime, state: dict) -> None:
    runtime.shared_prefix_len = int(state.get("shared_prefix_len", 0))
    runtime.num_told_global = int(state.get("num_told_global", 0))
    runtime.allocated_num_arms = state.get("allocated_num_arms")
    runtime.proposal_per_region = state.get("proposal_per_region")
    pending = state.get("last_region_indices")
    runtime.last_region_indices = None if pending is None else [int(i) for i in pending]


def _validate_runtime_load(
    *,
    runtime,
    data: list,
) -> None:
    if runtime.num_told_global < 0:
        raise ValueError("num_told_global must be >= 0")
    if runtime.shared_prefix_len < 0:
        raise ValueError("shared_prefix_len must be >= 0")
    if runtime.shared_prefix_len > runtime.num_told_global:
        raise ValueError("shared_prefix_len cannot exceed num_told_global")
    if len(data) < runtime.num_told_global:
        raise ValueError("load_state_dict received less data than num_told_global")


def _load_shared_region_data(*, designer: MultiTurboENNDesigner, runtime, data: list) -> None:
    runtime.region_data = [list(data[: runtime.num_told_global]) for _ in range(designer._num_regions)]
    runtime.region_assignments = []


def _load_independent_region_data(
    *,
    designer: MultiTurboENNDesigner,
    runtime,
    state: dict,
    data: list,
) -> None:
    assignments = state.get("region_assignments")
    if not isinstance(assignments, list):
        raise ValueError("Missing region_assignments for independent strategy")
    runtime.region_assignments = [int(i) for i in assignments]
    expected = int(runtime.num_told_global) - int(runtime.shared_prefix_len)
    if len(runtime.region_assignments) != expected:
        raise ValueError("region_assignments length does not match num_told_global/shared_prefix_len")
    runtime.region_data = [list(data[: runtime.shared_prefix_len]) for _ in range(designer._num_regions)]
    for offset, region_idx in enumerate(runtime.region_assignments):
        if region_idx < 0 or region_idx >= designer._num_regions:
            raise ValueError(f"Invalid region index {region_idx}")
        idx = runtime.shared_prefix_len + offset
        runtime.region_data[region_idx].append(data[idx])


def _load_region_child_states(*, designer: MultiTurboENNDesigner, state: dict, runtime) -> None:
    region_states = state.get("region_states") or []
    for idx, child_designer in enumerate(designer._designers):
        if idx >= len(region_states):
            continue
        child_designer.load_state_dict(region_states[idx], data=runtime.region_data[idx])


def load_multi_state(designer: MultiTurboENNDesigner, state: dict, data: list) -> None:
    designer._init_regions([], 1)
    _restore_rng_states(designer, state)

    runtime = designer._state
    _load_runtime_fields(runtime, state)
    _validate_runtime_load(runtime=runtime, data=data)

    if designer._strategy == "shared_data":
        _load_shared_region_data(designer=designer, runtime=runtime, data=data)
    else:
        _load_independent_region_data(designer=designer, runtime=runtime, state=state, data=data)
    _load_region_child_states(designer=designer, state=state, runtime=runtime)


def _tell_new_data_if_any(designer: MultiTurboENNDesigner, data: list) -> None:
    state = designer._state
    if len(data) <= state.num_told_global:
        return
    new_data = data[state.num_told_global :]
    if designer._strategy == "shared_data":
        for region_data in state.region_data:
            region_data.extend(new_data)
        state.last_region_indices = None
    else:
        pending = state.last_region_indices
        if pending is None:
            raise RuntimeError("Missing region assignments for new data")
        if len(new_data) > len(pending):
            raise RuntimeError("More new data than previous proposals")
        assign_now = pending[: len(new_data)]
        for datum, region_idx in zip(new_data, assign_now, strict=True):
            if region_idx < 0 or region_idx >= designer._num_regions:
                raise RuntimeError(f"Invalid region index {region_idx}")
            state.region_data[region_idx].append(datum)
            state.region_assignments.append(int(region_idx))
        remaining = pending[len(new_data) :]
        state.last_region_indices = remaining if remaining else None
    state.num_told_global = len(data)
