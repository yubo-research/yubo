from __future__ import annotations

import numpy as np


def test_mezo_tracks_legacy_real_and_predicted_bests_separately():
    from ops.vec_uhd import _new_mezo_state, _track_mezo_best

    objective = _Objective(dim=2)
    state = _new_mezo_state(objective)

    real_x = np.asarray([1.0, 0.0])
    state.last_mu = 1.0
    state.last_imputed = False
    _track_mezo_best(state, real_x)

    pred_x = np.asarray([2.0, 0.0])
    state.last_mu = 2.0
    state.last_imputed = True
    _track_mezo_best(state, pred_x)

    assert state.y_best == 2.0
    assert state.y_best_real == 1.0
    assert state.y_best_pred == 2.0
    np.testing.assert_allclose(state.best_x, pred_x)
    np.testing.assert_allclose(state.best_x_real, real_x)


def test_bszo_tracks_real_best_after_predicted_best():
    from ops.vec_uhd import _new_bszo_state, _track_bszo_best

    objective = _Objective(dim=2)
    state = _new_bszo_state(objective)

    pred_x = np.asarray([2.0, 0.0])
    state.last_imputed = True
    _track_bszo_best(state, pred_x, 2.0)

    real_x = np.asarray([1.0, 0.0])
    state.last_imputed = False
    _track_bszo_best(state, real_x, 1.0)

    assert state.y_best == 2.0
    assert state.y_best_real == 1.0
    assert state.y_best_pred == 2.0
    np.testing.assert_allclose(state.best_x, pred_x)
    np.testing.assert_allclose(state.best_x_real, real_x)


def test_best_source_suffix_only_when_enabled():
    from ops.vec_uhd import _format_source_best_suffix, _new_mezo_state, _track_mezo_best

    objective = _Objective(dim=1)
    state = _new_mezo_state(objective)
    state.last_mu = 1.0
    _track_mezo_best(state, np.asarray([1.0]))

    assert _format_source_best_suffix(state, False) == ""
    assert _format_source_best_suffix(state, True) == " y_best_real = 1.0000 y_best_pred = N/A"


class _Objective:
    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self.x0 = np.zeros(self.dim, dtype=np.float64)
