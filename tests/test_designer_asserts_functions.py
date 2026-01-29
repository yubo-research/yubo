import numpy as np
import pytest


class MockTrajectory:
    def __init__(self, rreturn):
        self.rreturn = rreturn


class MockDatum:
    def __init__(self, rreturn):
        self.trajectory = MockTrajectory(rreturn)


def test_assert_scalar_rreturn_scalar():
    from optimizer.designer_asserts import assert_scalar_rreturn

    data = [MockDatum(1.0), MockDatum(2.0)]
    assert_scalar_rreturn(data)


def test_assert_scalar_rreturn_vector_fails():
    from optimizer.designer_asserts import assert_scalar_rreturn

    data = [MockDatum(np.array([1.0, 2.0]))]
    with pytest.raises(AssertionError):
        assert_scalar_rreturn(data)


def test_assert_scalar_rreturn_empty():
    from optimizer.designer_asserts import assert_scalar_rreturn

    data = []
    assert_scalar_rreturn(data)
