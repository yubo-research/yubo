import numpy as np
import pytest


def test_mk_2d_single():
    from problems.benchmark_util import mk_2d

    x = np.array([5.0])
    y = mk_2d(x)
    assert y.shape == (1,) or y.shape == (2,)


@pytest.mark.parametrize(
    "fn_name,expected_shape",
    [
        ("mk_2d", (2,)),
        ("mk_4d", (4,)),
    ],
)
def test_mk_multiple(fn_name, expected_shape):
    from problems import benchmark_util

    fn = getattr(benchmark_util, fn_name)
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = fn(x)
    assert y.shape == expected_shape
