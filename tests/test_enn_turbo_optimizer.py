import numpy as np
from enn.turbo.types.appendable_array import AppendableArray

from optimizer.enn_turbo_optimizer import TurboOptimizer


def _appendable(rows: list[list[float]]) -> AppendableArray:
    arr = AppendableArray()
    for row in rows:
        arr.append(np.asarray(row, dtype=float))
    return arr


def test_trim_trailing_obs_clamps_prev_num_obs_for_tr_state():
    opt = object.__new__(TurboOptimizer)
    opt._x_obs = _appendable([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]])
    opt._y_obs = _appendable([[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]])
    opt._y_tr_list = [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]]
    opt._yvar_obs = AppendableArray()
    opt._trailing_obs = 4
    opt._incumbent_idx = 0

    class _TRState:
        prev_num_obs = 6

    opt._tr_state = _TRState()

    opt._trim_trailing_obs()

    assert len(opt._y_obs) <= 4
    assert int(opt._tr_state.prev_num_obs) == len(opt._y_obs)
