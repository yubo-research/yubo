"""Test that policy_tag is properly threaded to Optimizer.

BUG: sample_1 creates Optimizer without passing policy_tag from problem.policy_tag.
This causes _init_ref_point to fail for multi-objective optimization because
self._policy_tag is None.
"""

from unittest.mock import MagicMock, patch


def test_sample_1_passes_policy_tag_to_optimizer():
    """Verify that sample_1 passes policy_tag to Optimizer.

    This test exposes a bug where sample_1 creates Optimizer without policy_tag,
    causing multi-objective optimization to fail in _init_ref_point.
    """
    from experiments.experiment_sampler import RunConfig, sample_1

    mock_policy = MagicMock()
    mock_policy.num_params.return_value = 10

    mock_env_runtime = MagicMock()
    mock_env_runtime.problem_seed = 42
    mock_env_runtime.env_name = "test_env"
    mock_env_runtime.frozen_noise = True

    mock_problem = MagicMock()
    mock_problem.env = mock_env_runtime
    mock_problem.policy_tag = "test-policy-tag"
    mock_problem.build_policy.return_value = mock_policy

    run_config = RunConfig(
        problem=mock_problem,
        opt_name="random",
        num_rounds=1,
        num_arms=1,
        num_denoise=None,
        num_denoise_passive=None,
        max_proposal_seconds=None,
        b_trace=False,
        trace_fn="/tmp/test_trace",
    )

    captured_kwargs = {}

    def mock_optimizer_init(self, collector, **kwargs):
        captured_kwargs.update(kwargs)
        self.best_policy = kwargs["policy"]
        self._t_0 = 0

    mock_trace_entry = MagicMock()
    mock_trace_entry.dt_prop = 0.1
    mock_trace_entry.dt_eval = 0.2
    mock_trace_entry.rreturn = 1.0
    mock_trace_entry.env_steps_iter = 0
    mock_trace_entry.env_steps_total = 0

    with patch("experiments.experiment_sampler.seed_all"):
        with patch("experiments.experiment_sampler.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.empty.return_value.device = "cpu"

            with patch("optimizer.optimizer.Optimizer") as MockOptimizer:
                mock_opt_instance = MagicMock()
                mock_opt_instance.collect_trace.return_value = iter([mock_trace_entry])
                mock_opt_instance.best_policy = mock_policy
                mock_opt_instance._t_0 = 0
                MockOptimizer.return_value = mock_opt_instance

                sample_1(run_config)

                call_kwargs = MockOptimizer.call_args.kwargs
                assert "policy_tag" in call_kwargs, (
                    "BUG: sample_1 does not pass policy_tag to Optimizer. This breaks multi-objective optimization in _init_ref_point."
                )
                assert call_kwargs["policy_tag"] == "test-policy-tag", f"Expected policy_tag='test-policy-tag', got {call_kwargs.get('policy_tag')}"
