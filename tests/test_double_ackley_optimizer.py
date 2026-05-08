import pytest

from tests.optimizer_trace_short import assert_short_optimizer_trace_finite


@pytest.mark.parametrize(
    "designer_name,num_arms",
    [
        ("sobol", 4),
        ("morbo-enn-fit/acq_type=ucb", 1),
    ],
)
def test_double_ackley_runs(designer_name, num_arms):
    assert_short_optimizer_trace_finite(
        "f:doubleackley-20d",
        designer_name=designer_name,
        noise_seed_0=17,
        num_arms=num_arms,
        policy_tag="pure-function",
    )
