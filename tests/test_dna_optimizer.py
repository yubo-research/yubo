from tests.optimizer_trace_short import assert_short_optimizer_trace_finite


def test_dna_runs_with_random_optimizer():
    assert_short_optimizer_trace_finite("dna", designer_name="random")
