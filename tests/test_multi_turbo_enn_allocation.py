import numpy as np
import pytest


class _FixedChild:
    def __init__(self, n_params: int, n_emit: int):
        self._n_params = int(n_params)
        self._n_emit = int(n_emit)

    def __call__(self, _data, _n):
        out = []
        for _ in range(self._n_emit):
            out.append(_Policy(np.zeros((self._n_params,), dtype=float)))
        return out


class _Policy:
    def __init__(self, params):
        self._params = np.asarray(params, dtype=float)

    def get_params(self):
        return self._params


def test_fixed_region_counts_split():
    from optimizer.multi_turbo_enn_allocation import fixed_region_counts

    counts = fixed_region_counts(num_arms=5, num_regions=2, arm_mode="split", per_region_fn=None)
    assert counts == [3, 2]


def test_allocated_proposal_count_sets_first_allocation():
    from optimizer.multi_turbo_enn_allocation import allocated_proposal_count

    proposal, allocated = allocated_proposal_count(
        num_arms=5,
        num_regions=2,
        pool_multiplier=2,
        allocated_num_arms=None,
        proposal_per_region=None,
    )
    assert allocated == 5
    assert proposal == 6


def test_propose_batch_collects_region_indices():
    from optimizer.multi_turbo_enn_allocation import propose_batch

    children = [_FixedChild(2, 1), _FixedChild(2, 2)]
    region_data = [[], []]
    batch = propose_batch(children, region_data, per_region=[1, 1])
    assert len(batch.policies) == 3
    assert batch.region_indices == [0, 1, 1]


def test_select_top_k_orders_by_score():
    from optimizer.multi_turbo_enn_allocation import select_top_k

    scores = np.array([1.0, 3.0, 2.0, 4.0], dtype=float)
    chosen = select_top_k(scores, k=2)
    assert list(chosen) == [3, 1]


def test_select_top_k_rejects_invalid_k():
    from optimizer.multi_turbo_enn_allocation import select_top_k

    scores = np.array([1.0, 2.0], dtype=float)
    with pytest.raises(ValueError, match="k must be > 0"):
        select_top_k(scores, k=0)
    with pytest.raises(ValueError, match="must be <= num_scores"):
        select_top_k(scores, k=3)


def test_propose_batch_rejects_length_mismatch():
    from optimizer.multi_turbo_enn_allocation import propose_batch

    with pytest.raises(ValueError, match="per_region length mismatch"):
        propose_batch([_FixedChild(2, 1)], [[]], per_region=[1, 1])


def test_proposal_batch_direct_construction():
    from optimizer.multi_turbo_enn_allocation import ProposalBatch

    batch = ProposalBatch(policies=[], region_indices=[])
    assert batch.policies == []
    assert batch.region_indices == []


def test_allocated_proposal_plan():
    from optimizer.multi_turbo_enn_allocation import (
        AllocatedProposalPlan,
        allocated_proposal_plan,
    )

    plan = allocated_proposal_plan(
        num_arms=6,
        num_regions=2,
        pool_multiplier=2,
        allocated_num_arms=None,
        proposal_per_region=None,
    )
    assert isinstance(plan, AllocatedProposalPlan)
    assert plan.allocated_num_arms == 6
    assert plan.proposal_per_region >= 1
    assert len(plan.per_region) == 2
