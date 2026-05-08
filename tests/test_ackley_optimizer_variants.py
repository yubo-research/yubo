import pytest

from tests.optimizer_trace_short import assert_short_optimizer_trace_finite


@pytest.mark.parametrize("designer_name", ["vecchia", "turbo-zero"])
def test_ackley_3d_runs_with_optimizer(designer_name):
    assert_short_optimizer_trace_finite("f:ackley-3d", designer_name=designer_name)
