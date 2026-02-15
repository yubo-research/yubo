from unittest.mock import patch

import numpy as np
from click.testing import CliRunner

from experiments.exp_uhd import cli
from optimizer.trajectories import Trajectory


def _make_trajectory(rreturn):
    return Trajectory(
        rreturn=rreturn,
        states=np.zeros((1, 1)),
        actions=np.zeros((1, 1)),
    )


def test_local_rl_runs():
    call_count = [0]

    def mock_collect(ec, pol, **kwargs):
        call_count[0] += 1
        return _make_trajectory(float(call_count[0]))

    runner = CliRunner()
    with patch("optimizer.trajectories.collect_trajectory", side_effect=mock_collect):
        result = runner.invoke(cli, ["local", "--env-tag", "lunar", "--num-rounds", "3"])

    assert result.exit_code == 0, result.output
    assert call_count[0] == 3
    assert "EVAL: i_iter = 0" in result.output
    assert "EVAL: i_iter = 2" in result.output


def test_local_mnist_runs():
    runner = CliRunner()
    result = runner.invoke(cli, ["local", "--env-tag", "mnist", "--num-rounds", "2"])

    assert result.exit_code == 0, result.output
    assert "EVAL: i_iter = 0" in result.output
    assert "EVAL: i_iter = 1" in result.output
