from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optimizer.designer_asserts import assert_scalar_rreturn

if TYPE_CHECKING:
    from optimizer.multi_turbo_enn_designer import MultiTurboENNDesigner


def _is_thompson_like(acq_type: str) -> bool:
    return acq_type in ("thompson", "draw")


def _score_scalar_posterior(
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    acq_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    mu = np.asarray(mu, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    if mu.shape != sigma.shape:
        raise RuntimeError(f"mu/sigma shape mismatch: {mu.shape} vs {sigma.shape}")
    if _is_thompson_like(acq_type):
        return mu + sigma * rng.normal(size=mu.shape[0])
    return mu + sigma


def _score_multi_posterior(
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    acq_type: str,
    scalarize_fn,
    rng: np.random.Generator,
    allow_random_fallback: bool,
) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if mu.ndim == 1:
        mu = mu.reshape(-1, 1)
    if sigma.ndim == 1:
        sigma = sigma.reshape(-1, 1)
    if mu.shape != sigma.shape:
        raise RuntimeError(f"mu/sigma shape mismatch: {mu.shape} vs {sigma.shape}")
    if _is_thompson_like(acq_type):
        y = mu + sigma * rng.normal(size=mu.shape)
    else:
        y = mu + sigma
    scalar = scalarize_fn(y)
    if scalar is None:
        if allow_random_fallback:
            return rng.random(size=(mu.shape[0],))
        raise RuntimeError("multi-objective scoring requires scalarize support")
    scalar = np.asarray(scalar, dtype=float).reshape(-1)
    if scalar.shape[0] != mu.shape[0]:
        raise RuntimeError(f"scalarized scores shape {scalar.shape} != {mu.shape[0]}")
    return scalar


def _fallback_region_scores(
    designer: MultiTurboENNDesigner,
    *,
    region_idx: int,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    best_datum = designer._designers[region_idx].best_datum()
    best_val = 0.0
    if best_datum is not None:
        best_val = float(best_datum.trajectory.get_decision_rreturn())
    return best_val + rng.random(size=(n,))


def _score_region_candidates(
    designer: MultiTurboENNDesigner,
    *,
    region_idx: int,
    x_region: np.ndarray,
) -> np.ndarray:
    mu_sigma = designer._predict_mu_sigma(region_idx, x_region)
    region_rng = designer._region_rngs[region_idx]
    if mu_sigma is None:
        return _fallback_region_scores(designer, region_idx=region_idx, n=x_region.shape[0], rng=region_rng)
    mu, sigma = mu_sigma
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if mu.ndim == 1:
        mu = mu.reshape(-1, 1)
    if sigma.ndim == 1:
        sigma = sigma.reshape(-1, 1)
    if mu.shape != sigma.shape:
        raise RuntimeError(f"mu/sigma shape mismatch: {mu.shape} vs {sigma.shape}")
    if mu.shape[0] != x_region.shape[0]:
        raise RuntimeError(f"mu rows {mu.shape[0]} != candidates {x_region.shape[0]}")
    if mu.shape[1] == 1:
        return _score_scalar_posterior(mu[:, 0], sigma[:, 0], acq_type=designer._acq_type, rng=region_rng)
    allow_random = len(designer._region_data[region_idx]) == 0
    return _score_multi_posterior(
        mu,
        sigma,
        acq_type=designer._acq_type,
        scalarize_fn=lambda y: designer._scalarize(region_idx, y),
        rng=region_rng,
        allow_random_fallback=allow_random,
    )


def score_multi_candidates(
    designer: MultiTurboENNDesigner,
    x_all: np.ndarray,
    region_indices: list[int],
) -> np.ndarray:
    scores = np.full((x_all.shape[0],), np.nan, dtype=float)
    per_region: list[list[int]] = [[] for _ in range(designer._num_regions)]
    for idx, region_idx in enumerate(region_indices):
        per_region[region_idx].append(idx)
    for region_idx, indices in enumerate(per_region):
        if not indices:
            continue
        idx_arr = np.asarray(indices, dtype=int)
        scores[idx_arr] = _score_region_candidates(designer, region_idx=region_idx, x_region=x_all[idx_arr])
    nan_mask = ~np.isfinite(scores)
    if np.any(nan_mask):
        scores[nan_mask] = designer._rng.random(size=int(np.sum(nan_mask)))
    return scores + designer._rng.uniform(0.0, 1e-12, size=scores.shape)


def _restore_root_rng(designer: MultiTurboENNDesigner, state: dict) -> None:
    rng_state = state.get("rng_state")
    if rng_state is not None:
        designer._rng.bit_generator.state = rng_state


def _restore_region_rngs(designer: MultiTurboENNDesigner, state: dict) -> None:
    region_rng_states = state.get("region_rng_states")
    if region_rng_states is None:
        return
    if len(region_rng_states) != len(designer._region_rngs):
        raise ValueError("region_rng_states has wrong length")
    for rng, st in zip(designer._region_rngs, region_rng_states, strict=True):
        rng.bit_generator.state = st


def _restore_meta(designer: MultiTurboENNDesigner, state: dict) -> None:
    designer._shared_prefix_len = int(state.get("shared_prefix_len", 0))
    designer._num_told_global = int(state.get("num_told_global", 0))
    designer._allocated_num_arms = state.get("allocated_num_arms")
    designer._proposal_per_region = state.get("proposal_per_region")
    designer._last_region_indices = state.get("last_region_indices")


def _validate_meta(designer: MultiTurboENNDesigner, data: list) -> None:
    if designer._num_told_global < 0:
        raise ValueError("num_told_global must be >= 0")
    if designer._shared_prefix_len < 0:
        raise ValueError("shared_prefix_len must be >= 0")
    if designer._shared_prefix_len > designer._num_told_global:
        raise ValueError("shared_prefix_len cannot exceed num_told_global")
    if len(data) < designer._num_told_global:
        raise ValueError("load_state_dict received less data than num_told_global")


def _restore_region_data(
    designer: MultiTurboENNDesigner,
    state: dict,
    *,
    data: list,
) -> None:
    if designer._strategy == "shared_data":
        designer._region_data = [list(data[: designer._num_told_global]) for _ in range(designer._num_regions)]
        designer._region_assignments = []
        return

    assignments = state.get("region_assignments")
    if not isinstance(assignments, list):
        raise ValueError("Missing region_assignments for independent strategy")
    designer._region_assignments = [int(i) for i in assignments]
    expected = int(designer._num_told_global) - int(designer._shared_prefix_len)
    if len(designer._region_assignments) != expected:
        raise ValueError("region_assignments length does not match num_told_global/shared_prefix_len")

    designer._region_data = [list(data[: designer._shared_prefix_len]) for _ in range(designer._num_regions)]
    for offset, region_idx in enumerate(designer._region_assignments):
        if region_idx < 0 or region_idx >= designer._num_regions:
            raise ValueError(f"Invalid region index {region_idx}")
        idx = designer._shared_prefix_len + offset
        designer._region_data[region_idx].append(data[idx])


def _restore_region_states(designer: MultiTurboENNDesigner, state: dict) -> None:
    region_states = state.get("region_states") or []
    for idx, child_designer in enumerate(designer._designers):
        if idx >= len(region_states):
            continue
        region_data = designer._region_data[idx]
        child_designer.load_state_dict(region_states[idx], data=region_data)


def load_multi_state(designer: MultiTurboENNDesigner, state: dict, data: list) -> None:
    _restore_root_rng(designer, state)
    designer._init_regions([], num_arms=1)
    _restore_region_rngs(designer, state)
    _restore_meta(designer, state)
    _validate_meta(designer, data)
    _restore_region_data(designer, state, data=data)
    _restore_region_states(designer, state)


def _tell_new_data_if_any(designer: MultiTurboENNDesigner, data: list) -> None:
    if len(data) <= designer._num_told_global:
        return
    new_data = data[designer._num_told_global :]
    if designer._strategy == "shared_data":
        designer._broadcast_new_data(new_data)
        designer._last_region_indices = None
    else:
        designer._assign_new_data(new_data)
    designer._num_told_global = len(data)


def _ensure_allocated_mode(designer: MultiTurboENNDesigner, *, num_arms: int) -> None:
    if designer._allocated_num_arms is None:
        designer._allocated_num_arms = num_arms
        base = int(np.ceil(num_arms / float(designer._num_regions)))
        designer._proposal_per_region = max(1, base * designer._pool_multiplier)
        return
    if num_arms != designer._allocated_num_arms:
        raise ValueError(f"allocated arm_mode expects fixed num_arms; got {num_arms}, expected {designer._allocated_num_arms}")


def _propose_allocated(
    designer: MultiTurboENNDesigner,
    *,
    proposal_per_region: int,
) -> tuple[list, list[int]]:
    policies_all: list = []
    region_indices: list[int] = []
    for region_idx, child_designer in enumerate(designer._designers):
        policies = child_designer(designer._region_data[region_idx], proposal_per_region)
        policies_all.extend(policies)
        region_indices.extend([region_idx] * len(policies))
    return policies_all, region_indices


def _select_allocated(
    designer: MultiTurboENNDesigner,
    *,
    policies_all: list,
    region_indices: list[int],
    num_arms: int,
) -> list:
    if not policies_all:
        designer._last_region_indices = []
        return []

    x_all = np.array([p.get_params() for p in policies_all], dtype=float)
    if x_all.shape[0] <= num_arms:
        designer._last_region_indices = region_indices
        return policies_all

    scores = designer._score_candidates(x_all, region_indices)
    chosen = np.argpartition(-scores, num_arms - 1)[:num_arms]
    chosen = chosen[np.argsort(-scores[chosen])]
    designer._last_region_indices = [region_indices[int(i)] for i in chosen]
    for region_idx in range(designer._num_regions):
        num_assigned = sum(1 for idx in designer._last_region_indices if idx == region_idx)
        designer._adjust_failure_tolerance(region_idx, num_assigned)
    return [policies_all[int(i)] for i in chosen]


def _call_allocated(
    designer: MultiTurboENNDesigner,
    *,
    num_arms: int,
) -> list:
    _ensure_allocated_mode(designer, num_arms=num_arms)
    proposal_per_region = int(designer._proposal_per_region or 0)
    if proposal_per_region <= 0:
        raise RuntimeError("proposal_per_region is not set")
    policies_all, region_indices = _propose_allocated(designer, proposal_per_region=proposal_per_region)
    return _select_allocated(designer, policies_all=policies_all, region_indices=region_indices, num_arms=num_arms)


def _call_fixed_arms(
    designer: MultiTurboENNDesigner,
    *,
    num_arms: int,
) -> list:
    per_region = designer._per_region_counts(num_arms)
    policies_all: list = []
    region_indices: list[int] = []
    for region_idx, (child_designer, n) in enumerate(zip(designer._designers, per_region, strict=True)):
        if n <= 0:
            continue
        policies = child_designer(designer._region_data[region_idx], n)
        policies_all.extend(policies)
        region_indices.extend([region_idx] * len(policies))
    designer._last_region_indices = region_indices
    return policies_all


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
