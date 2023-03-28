import numpy as np


def _test(problem_name, num_dim, x, y):
    from problems.pure_functions import make

    fn = make(f"f:{problem_name}-{num_dim}d", seed=17)
    y_check = fn.step(x)[1]
    assert abs(y_check - y) < 1e-6, (x, y_check, y)


def test_michalewicz():
    _test(problem_name="michalewicz", num_dim=2, x=np.array([0.3, 0.3]), y=0.17602377327570404)
