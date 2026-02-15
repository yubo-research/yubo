import numpy as np

from turbo_m_ref.turbo_m import TurboM
from turbo_m_ref.utils import turbo_adjust_length


def test_turbo_m_init():
    lb = np.zeros(2)
    ub = np.ones(2)
    tm = TurboM(
        f=lambda x: np.sum(x**2),
        lb=lb,
        ub=ub,
        n_init=4,
        max_evals=20,
        n_trust_regions=2,
        verbose=False,
    )
    assert tm.n_trust_regions == 2


def test_turbo_adjust_length():
    class MockOpt:
        length = 0.5
        succcount = 0
        failcount = 0
        succtol = 3
        failtol = 3
        length_max = 2.0
        _fX = np.array([1.0, 2.0])

    opt = MockOpt()
    turbo_adjust_length(opt, np.array([0.5]))  # improvement
    assert opt.succcount == 1
