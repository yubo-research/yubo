from __future__ import annotations

from unittest.mock import MagicMock


def make_mock_problem_for_sampler():
    mock_env = MagicMock()
    mock_env.env_name = "test_env"
    mock_env.problem_seed = 42
    mock_problem = MagicMock()
    mock_problem.env = mock_env
    mock_problem.build_policy.return_value = MagicMock()
    return mock_problem
