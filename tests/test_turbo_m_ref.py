import numpy as np


def test_turbo1_init():
    from turbo_m_ref.turbo_1 import Turbo1

    def dummy_f(x):
        return -np.sum(x**2, axis=1)

    lb = np.zeros(2)
    ub = np.ones(2)

    turbo = Turbo1(
        f=dummy_f,
        lb=lb,
        ub=ub,
        n_init=4,
        max_evals=10,
        batch_size=1,
        verbose=False,
    )
    assert turbo is not None


def test_turbo1_optimize():
    from turbo_m_ref.turbo_1 import Turbo1

    def dummy_f(x):
        return -np.sum(x**2, axis=1)

    lb = np.zeros(2)
    ub = np.ones(2)

    turbo = Turbo1(
        f=dummy_f,
        lb=lb,
        ub=ub,
        n_init=4,
        max_evals=6,
        batch_size=1,
        verbose=False,
    )
    turbo.optimize()
    assert len(turbo.X) >= 4
    assert len(turbo.fX) >= 4


def test_turbo1_ask_tell_init():
    from turbo_m_ref.turbo_1_ask_tell import Turbo1

    x_bounds = np.array([[0.0, 1.0], [0.0, 1.0]])

    turbo = Turbo1(
        x_bounds=x_bounds,
        n_init=4,
        batch_size=1,
        verbose=False,
    )
    assert turbo is not None


def test_turbo1_ask_tell_ask():
    from turbo_m_ref.turbo_1_ask_tell import Turbo1

    x_bounds = np.array([[0.0, 1.0], [0.0, 1.0]])

    turbo = Turbo1(
        x_bounds=x_bounds,
        n_init=4,
        batch_size=1,
        verbose=False,
    )
    x = turbo.ask()
    assert x.shape == (2,)  # Returns 1D array for batch_size=1


def test_turbo_m_init():
    from turbo_m_ref.turbo_m import TurboM

    def dummy_f(x):
        return -np.sum(x**2, axis=1)

    lb = np.zeros(2)
    ub = np.ones(2)

    turbo = TurboM(
        f=dummy_f,
        lb=lb,
        ub=ub,
        n_init=4,
        max_evals=10,
        n_trust_regions=2,
        batch_size=1,
        verbose=False,
    )
    assert turbo is not None


def test_turbo1_ask_tell_tell():
    from turbo_m_ref.turbo_1_ask_tell import Turbo1

    x_bounds = np.array([[0.0, 1.0], [0.0, 1.0]])

    turbo = Turbo1(
        x_bounds=x_bounds,
        n_init=4,
        batch_size=1,
        verbose=False,
    )

    # Ask and tell several times
    for _ in range(4):
        x = turbo.ask()
        y = float(-np.sum(x**2))  # Must be a scalar
        turbo.tell(y, x)  # y first, then x

    # Should have recorded data
    assert len(turbo.X) == 4
    assert len(turbo.fX) == 4
