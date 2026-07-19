import pytest

from tests.optimizer_trace_short import assert_short_optimizer_trace_finite


def test_dna_runs_with_random_optimizer():
    try:
        assert_short_optimizer_trace_finite("dna", designer_name="random")
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"dna/numba unavailable: {exc}")
