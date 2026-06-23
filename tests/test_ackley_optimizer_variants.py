import pytest

from tests.optimizer_trace_short import assert_short_optimizer_trace_finite


@pytest.mark.parametrize(
    "designer_name",
    [
        "vecchia",
        "turbo-zero",
        "turbo-mars-ucb/num_init=2/num_candidates=32/max_terms=8/num_bootstrap=2/feature_screen=3/active_samples=8",
        "turbo-bmars-ucb/num_init=2/num_candidates=32/max_terms=6/feature_screen=3/mcmc_steps=4/mcmc_burn_in=4/mcmc_num_models=2/mcmc_pool_size=6",
    ],
)
def test_ackley_3d_runs_with_optimizer(designer_name):
    assert_short_optimizer_trace_finite("f:ackley-3d", designer_name=designer_name)
