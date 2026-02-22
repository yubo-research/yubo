from __future__ import annotations

import numpy as np
from attrs import define


@define(frozen=True)
class ProposalBatch:
    policies: list
    region_indices: list[int]


@define(frozen=True)
class AllocatedProposalPlan:
    allocated_num_arms: int
    proposal_per_region: int
    per_region: list[int]


def propose_batch(child_designers: list, region_data: list[list], *, per_region: list[int]) -> ProposalBatch:
    if len(child_designers) != len(region_data):
        raise ValueError("child_designers and region_data length mismatch")
    if len(per_region) != len(child_designers):
        raise ValueError("per_region length mismatch")
    policies_all: list = []
    region_indices: list[int] = []
    for region_idx, (child_designer, n) in enumerate(zip(child_designers, per_region, strict=True)):
        if n <= 0:
            continue
        policies = child_designer(region_data[region_idx], int(n))
        policies_all.extend(policies)
        region_indices.extend([region_idx] * len(policies))
    return ProposalBatch(policies=policies_all, region_indices=region_indices)


def fixed_region_counts(
    *,
    num_arms: int,
    num_regions: int,
    arm_mode: str,
    per_region_fn,
) -> list[int]:
    if callable(per_region_fn):
        return list(per_region_fn(num_arms))
    if arm_mode == "per_region":
        return [num_arms] * num_regions
    if num_arms < num_regions:
        raise ValueError(f"num_arms={num_arms} must be >= num_regions={num_regions} when arm_mode='split'")
    base = num_arms // num_regions
    remainder = num_arms % num_regions
    return [base + (1 if i < remainder else 0) for i in range(num_regions)]


def allocated_proposal_count(
    *,
    num_arms: int,
    num_regions: int,
    pool_multiplier: int,
    allocated_num_arms: int | None,
    proposal_per_region: int | None,
) -> tuple[int, int]:
    if allocated_num_arms is None:
        allocated_num_arms = num_arms
        base = int(np.ceil(num_arms / float(num_regions)))
        proposal_per_region = max(1, base * pool_multiplier)
    elif num_arms != allocated_num_arms:
        raise ValueError(f"allocated arm_mode expects fixed num_arms; got {num_arms}, expected {allocated_num_arms}")
    proposal_per_region = int(proposal_per_region or 0)
    if proposal_per_region <= 0:
        raise RuntimeError("proposal_per_region is not set")
    return proposal_per_region, allocated_num_arms


def allocated_proposal_plan(
    *,
    num_arms: int,
    num_regions: int,
    pool_multiplier: int,
    allocated_num_arms: int | None,
    proposal_per_region: int | None,
) -> AllocatedProposalPlan:
    proposal, allocated = allocated_proposal_count(
        num_arms=num_arms,
        num_regions=num_regions,
        pool_multiplier=pool_multiplier,
        allocated_num_arms=allocated_num_arms,
        proposal_per_region=proposal_per_region,
    )
    return AllocatedProposalPlan(
        allocated_num_arms=allocated,
        proposal_per_region=proposal,
        per_region=[proposal] * int(num_regions),
    )


def select_top_k(scores: np.ndarray, *, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be > 0")
    if k > scores.shape[0]:
        raise ValueError(f"k={k} must be <= num_scores={scores.shape[0]}")
    chosen = np.argpartition(-scores, k - 1)[:k]
    return chosen[np.argsort(-scores[chosen])]
