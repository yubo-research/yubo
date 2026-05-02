import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from experiments.experiment_sampler import (
    ExperimentConfig,
    RunConfig,
    count_local_trace_jobs,
    extract_trace_fns,
    mk_replicates,
    prep_args_1,
    prep_d_args,
    true_false,
)


def test_true_false_true():
    assert true_false("true") is True
    assert true_false("True") is True
    assert true_false("TRUE") is True
    assert true_false("t") is True
    assert true_false("T") is True


def test_true_false_false():
    assert true_false("false") is False
    assert true_false("False") is False
    assert true_false("FALSE") is False
    assert true_false("f") is False
    assert true_false("F") is False


def test_true_false_invalid():
    with pytest.raises(AssertionError):
        true_false("yes")
    with pytest.raises(AssertionError):
        true_false("1")


def _make_mock_problem():
    """Create a mock Problem with .env property and .build_policy() method."""
    mock_env = MagicMock()
    mock_env.env_name = "test_env"
    mock_env.problem_seed = 42
    mock_problem = MagicMock()
    mock_problem.env = mock_env
    mock_problem.build_policy.return_value = MagicMock()
    return mock_problem


def test_extract_trace_fns():
    mock_problem = _make_mock_problem()
    run_configs = [
        RunConfig(
            problem=mock_problem,
            opt_name="ucb",
            num_rounds=10,
            num_arms=5,
            num_denoise=None,
            num_denoise_passive=None,
            max_proposal_seconds=None,
            b_trace=True,
            trace_fn="/path/a",
        ),
        RunConfig(
            problem=mock_problem,
            opt_name="ei",
            num_rounds=10,
            num_arms=5,
            num_denoise=None,
            num_denoise_passive=None,
            max_proposal_seconds=None,
            b_trace=True,
            trace_fn="/path/b",
        ),
    ]
    trace_fns = extract_trace_fns(run_configs)
    assert trace_fns == ["/path/a", "/path/b"]


def test_prep_args_1():
    result = prep_args_1(
        results_dir="/results",
        exp_dir="exp1",
        problem="f:ackley-10d",
        opt="ucb",
        num_arms=5,
        num_replications=3,
        num_rounds=10,
        noise=None,
        num_denoise=100,
        policy_tag="pure-function",
    )
    assert isinstance(result, ExperimentConfig)
    assert result.exp_dir == "/results/exp1"
    assert result.env_tag == "f:ackley-10d"
    assert result.opt_name == "ucb"
    assert result.num_arms == 5
    assert result.num_reps == 3
    assert result.num_rounds == 10
    assert result.num_denoise == 100
    assert result.policy_tag == "pure-function"


def test_prep_d_args():
    results = prep_d_args(
        results_dir="/results",
        exp_dir="exp1",
        funcs=["ackley", "sphere"],
        dims=[5, 10],
        num_arms=4,
        num_replications=2,
        opts=["ucb", "ei"],
        noises=[None],
        num_rounds=5,
        func_category="f",
        num_denoise=None,
        policy_tag="pure-function",
    )
    assert len(results) == 2 * 2 * 2 * 1
    assert all(isinstance(r, ExperimentConfig) for r in results)
    assert results[0].env_tag == "f:ackley-5d"
    assert results[0].opt_name == "ucb"
    assert results[0].policy_tag == "pure-function"


@patch("experiments.experiment_sampler.build_problem")
@patch("experiments.experiment_sampler.data_is_done")
def test_mk_replicates(mock_data_is_done, mock_build_problem):
    mock_data_is_done.return_value = False
    mock_problem = _make_mock_problem()
    mock_build_problem.return_value = mock_problem

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExperimentConfig(
            exp_dir=tmpdir,
            env_tag="f:ackley-10d",
            opt_name="ucb",
            num_arms=5,
            num_rounds=10,
            num_reps=3,
            num_denoise=100,
            b_trace=True,
            policy_tag="pure-function",
        )
        results = mk_replicates(config)

    assert len(results) == 3
    assert all(isinstance(r, RunConfig) for r in results)
    assert results[0].opt_name == "ucb"
    assert results[0].num_arms == 5
    assert results[0].num_rounds == 10
    assert results[0].num_denoise == 100
    assert results[0].b_trace is True
    assert results[0].problem == mock_problem


def test_count_local_trace_jobs_empty():
    assert count_local_trace_jobs([]) == (0, 0, 0)


def test_count_local_trace_jobs_one_done_one_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExperimentConfig(
            exp_dir=tmpdir,
            env_tag="f:ackley-10d",
            opt_name="ucb",
            num_arms=5,
            num_rounds=10,
            num_reps=2,
            num_denoise=None,
            b_trace=True,
        )
        out_dir = config.to_dir_name()
        trace0 = os.path.join(out_dir, "traces", "00000")
        os.makedirs(os.path.dirname(trace0), exist_ok=True)
        with open(trace0, "wb") as f:
            f.write(b"x\nDONE\n")
        done, left, total = count_local_trace_jobs([config])
        assert total == 2
        assert done == 1
        assert left == 1


@patch("experiments.experiment_sampler.build_problem")
@patch("experiments.experiment_sampler.data_is_done")
def test_mk_replicates_skips_done(mock_data_is_done, mock_build_problem):
    mock_data_is_done.return_value = True
    mock_problem = _make_mock_problem()
    mock_build_problem.return_value = mock_problem

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExperimentConfig(
            exp_dir=tmpdir,
            env_tag="f:ackley-10d",
            opt_name="ucb",
            num_arms=5,
            num_rounds=10,
            num_reps=3,
            num_denoise=None,
            policy_tag="pure-function",
        )
        results = mk_replicates(config)

    assert len(results) == 0


def test_mk_replicates_creates_out_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("experiments.experiment_sampler.build_problem") as mock_build_problem:
            with patch("experiments.experiment_sampler.data_is_done") as mock_data_is_done:
                mock_data_is_done.return_value = False
                mock_build_problem.return_value = _make_mock_problem()

                config = ExperimentConfig(
                    exp_dir=tmpdir,
                    env_tag="f:test-5d",
                    opt_name="random",
                    num_arms=2,
                    num_rounds=5,
                    num_reps=1,
                    num_denoise=None,
                    policy_tag="pure-function",
                )
                mk_replicates(config)

                expected_dir = config.to_dir_name()
                assert os.path.isdir(expected_dir)
                assert os.path.isfile(f"{expected_dir}/config.json")


def test_experiment_config_to_dir_name():
    config = ExperimentConfig(
        exp_dir="/results/exp1",
        env_tag="f:ackley-10d",
        opt_name="ucb",
        num_arms=5,
        num_rounds=10,
        num_reps=3,
        num_denoise=100,
        policy_tag="pure-function",
    )
    dir_name = config.to_dir_name()
    assert dir_name.startswith("/results/exp1/")
    assert len(dir_name.split("/")[-1]) == 8


def test_experiment_config_to_dir_name_legacy():
    config = ExperimentConfig(
        exp_dir="/results/exp1",
        env_tag="f:ackley-10d",
        opt_name="ucb",
        num_arms=5,
        num_rounds=10,
        num_reps=3,
        num_denoise=100,
        policy_tag="pure-function",
    )
    expected = "/results/exp1/env=f:ackley-10d--opt_name=ucb--num_arms=5--num_rounds=10--num_reps=3--num_denoise=100"
    assert config.to_dir_name_legacy() == expected


def test_experiment_config_to_dict():
    config = ExperimentConfig(
        exp_dir="/results/exp1",
        env_tag="f:ackley-10d",
        opt_name="ucb",
        num_arms=5,
        num_rounds=10,
        num_reps=3,
        num_denoise=100,
        policy_tag="pure-function",
    )
    d = config.to_dict()
    assert d["exp_dir"] == "/results/exp1"
    assert d["env_tag"] == "f:ackley-10d"
    assert d["opt_name"] == "ucb"
    assert d["num_arms"] == 5
    assert d["num_rounds"] == 10
    assert d["num_reps"] == 3
    assert d["num_denoise"] == 100
    assert d["policy_tag"] == "pure-function"


def test_experiment_config_from_dict():
    d = {
        "exp_dir": "/results/exp1",
        "env_tag": "f:ackley-10d",
        "opt_name": "ucb",
        "num_arms": "5",
        "num_rounds": "10",
        "num_reps": "3",
        "num_denoise": "100",
        "b_trace": "true",
        "policy_tag": "pure-function",
    }
    config = ExperimentConfig.from_dict(d)
    assert config.exp_dir == "/results/exp1"
    assert config.env_tag == "f:ackley-10d"
    assert config.opt_name == "ucb"
    assert config.num_arms == 5
    assert config.num_rounds == 10
    assert config.num_reps == 3
    assert config.num_denoise == 100
    assert config.b_trace is True
    assert config.policy_tag == "pure-function"


def test_experiment_config_from_dict_none_denoise():
    d = {
        "exp_dir": "/results/exp1",
        "env_tag": "f:ackley-10d",
        "opt_name": "ucb",
        "num_arms": 5,
        "num_rounds": 10,
        "num_reps": 3,
        "num_denoise": "None",
        "policy_tag": "pure-function",
    }
    config = ExperimentConfig.from_dict(d)
    assert config.num_denoise is None


def test_experiment_config_from_dict_total_timesteps_only():
    d = {
        "exp_dir": "/results/exp1",
        "env_tag": "f:ackley-10d",
        "opt_name": "ucb",
        "num_arms": 5,
        "total_timesteps": 123456,
        "num_reps": 3,
        "policy_tag": "pure-function",
    }
    config = ExperimentConfig.from_dict(d)
    assert config.total_timesteps == 123456
    assert config.num_rounds is None
