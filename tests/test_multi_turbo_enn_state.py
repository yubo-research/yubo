from optimizer.multi_turbo_enn_state import (
    RegionToleranceTarget,
    SelectionCommit,
    commit_selection,
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
