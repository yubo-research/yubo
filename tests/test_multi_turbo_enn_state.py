import numpy as np
import pytest

from optimizer.multi_turbo_enn_state import (
    RegionToleranceTarget,
    SelectionCommit,
    commit_selection,
    load_multi_state,
)


def _dummy_designer():
    state = type(
        "S",
        (),
        {
            "last_region_indices": None,
            "allocated_num_arms": None,
            "proposal_per_region": None,
        },
    )()
    return type("D", (), {"_state": state})()


def test_commit_selection_updates_runtime_fields_and_pending_indices():
    designer = _dummy_designer()
    commit_selection(
        designer,
        SelectionCommit(
            pending_region_indices=[0, 1],
            allocated_num_arms=4,
            proposal_per_region=8,
        ),
    )
    assert designer._state.allocated_num_arms == 4
    assert designer._state.proposal_per_region == 8
    assert designer._state.last_region_indices == [0, 1]

    commit_selection(
        designer,
        SelectionCommit(
            pending_region_indices=[1, 2],
        ),
    )
    assert designer._state.last_region_indices == [0, 1, 1, 2]


def test_commit_selection_updates_failure_tolerances_from_explicit_targets():
    designer = _dummy_designer()
    tol_by_region: dict[int, int] = {}

    targets = [
        RegionToleranceTarget(
            num_dim=8,
            set_failure_tolerance=lambda tol: tol_by_region.__setitem__(0, tol),
        ),
        RegionToleranceTarget(
            num_dim=4,
            set_failure_tolerance=lambda tol: tol_by_region.__setitem__(1, tol),
        ),
        RegionToleranceTarget(
            num_dim=0,
            set_failure_tolerance=lambda tol: tol_by_region.__setitem__(2, tol),
        ),
    ]

    commit_selection(
        designer,
        SelectionCommit(
            pending_region_indices=[0, 0, 1, 2],
            update_failure_tolerances=True,
            tolerance_targets=targets,
        ),
    )

    assert tol_by_region[0] == 4
    assert tol_by_region[1] == 4
    assert 2 not in tol_by_region


def _make_loader_designer(*, strategy: str, num_regions: int = 2):
    class _Child:
        def __init__(self):
            self.loaded = []

        def load_state_dict(self, state, data):
            self.loaded.append((state, list(data)))

    class _Designer:
        pass

    designer = _Designer()
    designer._strategy = strategy
    designer._num_regions = int(num_regions)
    designer._rng = np.random.default_rng(10)
    designer._region_rngs = [np.random.default_rng(100 + i) for i in range(designer._num_regions)]
    designer._designers = [_Child() for _ in range(designer._num_regions)]

    def _init_regions(_data, _num_arms):
        state = type("S", (), {})()
        state.region_data = []
        state.shared_prefix_len = 0
        state.region_assignments = []
        state.last_region_indices = None
        state.num_told_global = 0
        state.allocated_num_arms = None
        state.proposal_per_region = None
        designer._state = state

    designer._init_regions = _init_regions
    return designer


def test_load_multi_state_shared_data_populates_runtime_and_children():
    designer = _make_loader_designer(strategy="shared_data", num_regions=2)
    base_rng = np.random.default_rng(123)
    region_rng_0 = np.random.default_rng(456)
    region_rng_1 = np.random.default_rng(789)

    state = {
        "rng_state": base_rng.bit_generator.state,
        "region_rng_states": [region_rng_0.bit_generator.state, region_rng_1.bit_generator.state],
        "shared_prefix_len": 1,
        "num_told_global": 2,
        "allocated_num_arms": 4,
        "proposal_per_region": 8,
        "last_region_indices": [0, 1],
        "region_states": [{"r": 0}, {"r": 1}],
    }
    data = ["d0", "d1", "d2"]

    load_multi_state(designer, state, data)

    assert designer._state.shared_prefix_len == 1
    assert designer._state.num_told_global == 2
    assert designer._state.allocated_num_arms == 4
    assert designer._state.proposal_per_region == 8
    assert designer._state.last_region_indices == [0, 1]
    assert designer._state.region_data == [["d0", "d1"], ["d0", "d1"]]
    assert designer._state.region_assignments == []
    assert len(designer._designers[0].loaded) == 1
    assert len(designer._designers[1].loaded) == 1


def test_load_multi_state_independent_validates_assignment_length():
    designer = _make_loader_designer(strategy="independent", num_regions=2)
    state = {
        "rng_state": np.random.default_rng(1).bit_generator.state,
        "region_rng_states": [np.random.default_rng(2).bit_generator.state, np.random.default_rng(3).bit_generator.state],
        "shared_prefix_len": 1,
        "num_told_global": 4,
        "region_assignments": [0, 1],
    }
    with pytest.raises(ValueError, match="region_assignments length"):
        load_multi_state(designer, state, ["a", "b", "c", "d"])


def test_load_multi_state_validates_region_rng_state_length():
    designer = _make_loader_designer(strategy="shared_data", num_regions=2)
    state = {
        "rng_state": np.random.default_rng(1).bit_generator.state,
        "region_rng_states": [np.random.default_rng(2).bit_generator.state],
        "shared_prefix_len": 0,
        "num_told_global": 0,
    }
    with pytest.raises(ValueError, match="wrong length"):
        load_multi_state(designer, state, [])
