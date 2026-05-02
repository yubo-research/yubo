import time
from unittest.mock import MagicMock, patch

import pytest

from experiments.experiment_sampler import (
    ExperimentConfig,
    RunConfig,
    _SampleResult,
    _scan_local_parallel,
    post_process,
    post_process_stdout,
    sample_1,
    sampler,
    scan_local,
)


def _make_mock_problem():
    """Create a mock Problem with .env property and .build_policy() method."""
    mock_env = MagicMock()
    mock_env.env_name = "test_env"
    mock_env.problem_seed = 42
    mock_problem = MagicMock()
    mock_problem.env = mock_env
    mock_problem.build_policy.return_value = MagicMock()
    return mock_problem


@patch("optimizer.optimizer.Optimizer")
@patch("experiments.experiment_sampler.seed_all")
@patch("experiments.experiment_sampler.torch")
def test_sample_1(mock_torch, mock_seed_all, mock_optimizer_class):
    mock_torch.cuda.is_available.return_value = False
    mock_torch.empty.return_value.device = "cpu"

    mock_policy = MagicMock()

    mock_trace_entry = MagicMock()
    mock_trace_entry.dt_prop = 0.1
    mock_trace_entry.dt_eval = 0.2
    mock_trace_entry.rreturn = 1.5

    mock_optimizer = MagicMock()
    mock_optimizer.collect_trace.return_value = iter([mock_trace_entry, mock_trace_entry])
    mock_optimizer_class.return_value = mock_optimizer

    mock_env_runtime = MagicMock()
    mock_env_runtime.problem_seed = 42
    mock_env_runtime.env_name = "test_env"

    mock_problem = MagicMock()
    mock_problem.env = mock_env_runtime
    mock_problem.build_policy.return_value = mock_policy

    run_config = RunConfig(
        problem=mock_problem,
        opt_name="ucb",
        num_rounds=2,
        num_arms=5,
        num_denoise=None,
        num_denoise_passive=None,
        max_proposal_seconds=100.0,
        b_trace=True,
        trace_fn="/path/to/trace",
    )
    result = sample_1(run_config)

    mock_seed_all.assert_called_once_with(42 + 27)
    mock_problem.build_policy.assert_called_once()
    mock_optimizer_class.assert_called_once()
    mock_optimizer.collect_trace.assert_called_once_with(
        designer_name="ucb",
        max_iterations=2,
        max_proposal_seconds=100.0,
        deadline=None,
        max_total_timesteps=None,
    )

    trace_lines = list(result.collector_trace)
    assert len(trace_lines) == 3
    assert "TRACE:" in trace_lines[0]
    assert "DONE" in trace_lines[2]

    assert len(result.trace_records) == 2
    assert result.trace_records[0].i_iter == 0
    assert result.trace_records[0].rreturn == 1.5
    assert result.stop_reason == "completed"


@patch("optimizer.optimizer.Optimizer")
@patch("experiments.experiment_sampler.seed_all")
@patch("experiments.experiment_sampler.torch")
def test_sample_1_no_trace(mock_torch, mock_seed_all, mock_optimizer_class):
    mock_torch.cuda.is_available.return_value = False
    mock_torch.empty.return_value.device = "cpu"

    mock_trace_entry = MagicMock()
    mock_trace_entry.dt_prop = 0.1
    mock_trace_entry.dt_eval = 0.2
    mock_trace_entry.rreturn = 1.5

    mock_optimizer = MagicMock()
    mock_optimizer.collect_trace.return_value = iter([mock_trace_entry])
    mock_optimizer_class.return_value = mock_optimizer

    mock_env_runtime = MagicMock()
    mock_env_runtime.problem_seed = 0
    mock_env_runtime.env_name = "test_env"

    mock_problem = MagicMock()
    mock_problem.env = mock_env_runtime
    mock_problem.build_policy.return_value = MagicMock()

    run_config = RunConfig(
        problem=mock_problem,
        opt_name="random",
        num_rounds=1,
        num_arms=1,
        num_denoise=None,
        num_denoise_passive=None,
        max_proposal_seconds=None,
        b_trace=False,
        trace_fn="/path/to/trace",
    )
    result = sample_1(run_config)

    trace_lines = list(result.collector_trace)
    assert len(trace_lines) == 1
    assert trace_lines[0] == "DONE"

    assert len(result.trace_records) == 1
    assert result.stop_reason == "completed"


@patch("optimizer.optimizer.Optimizer")
@patch("experiments.experiment_sampler.seed_all")
@patch("experiments.experiment_sampler.torch")
def test_sample_1_total_timesteps_budget(mock_torch, mock_seed_all, mock_optimizer_class):
    mock_torch.cuda.is_available.return_value = False
    mock_torch.empty.return_value.device = "cpu"

    mock_trace_entry = MagicMock()
    mock_trace_entry.dt_prop = 0.1
    mock_trace_entry.dt_eval = 0.2
    mock_trace_entry.rreturn = 1.5
    mock_trace_entry.env_steps_iter = 17
    mock_trace_entry.env_steps_total = 17

    mock_optimizer = MagicMock()
    mock_optimizer.collect_trace.return_value = iter([mock_trace_entry])
    mock_optimizer_class.return_value = mock_optimizer

    mock_env_runtime = MagicMock()
    mock_env_runtime.problem_seed = 0
    mock_env_runtime.env_name = "test_env"

    mock_problem = MagicMock()
    mock_problem.env = mock_env_runtime
    mock_problem.build_policy.return_value = MagicMock()

    run_config = RunConfig(
        problem=mock_problem,
        opt_name="random",
        num_rounds=None,
        total_timesteps=500,
        num_arms=1,
        num_denoise=None,
        num_denoise_passive=None,
        max_proposal_seconds=None,
        b_trace=False,
        trace_fn="/path/to/trace",
    )
    sample_1(run_config)

    call_kwargs = mock_optimizer.collect_trace.call_args.kwargs
    assert call_kwargs["max_iterations"] > 10**6
    assert call_kwargs["max_total_timesteps"] == 500


@patch("optimizer.optimizer.Optimizer")
@patch("problems.env_conf.default_policy")
@patch("experiments.experiment_sampler.seed_all")
@patch("experiments.experiment_sampler.torch")
def test_sample_1_stop_reason_timesteps_with_deadline(mock_torch, mock_seed_all, mock_default_policy, mock_optimizer_class):
    """Regression test: verify stop_reason is 'completed' when timesteps budget exhausted.

    When max_total_timesteps causes early termination but the deadline was not hit,
    stop_reason should be 'completed' (not 'deadline'). This test verifies the fix
    is in place: stop_reason is only set to 'deadline' when time.time() >= deadline.
    """
    mock_torch.cuda.is_available.return_value = False
    mock_torch.empty.return_value.device = "cpu"
    mock_default_policy.return_value = MagicMock()

    mock_trace_entry = MagicMock()
    mock_trace_entry.dt_prop = 0.1
    mock_trace_entry.dt_eval = 0.2
    mock_trace_entry.rreturn = 1.5
    mock_trace_entry.env_steps_iter = 100
    mock_trace_entry.env_steps_total = 100

    mock_optimizer = MagicMock()
    mock_optimizer.collect_trace.return_value = iter([mock_trace_entry])
    mock_optimizer_class.return_value = mock_optimizer

    mock_env_conf = MagicMock()
    mock_env_conf.problem_seed = 0
    mock_env_conf.env_name = "test_env"

    run_config = RunConfig(
        env_conf=mock_env_conf,
        opt_name="random",
        num_rounds=100,
        total_timesteps=50,
        num_arms=1,
        num_denoise=None,
        num_denoise_passive=None,
        max_proposal_seconds=None,
        b_trace=False,
        trace_fn="/path/to/trace",
        deadline=time.time() + 3600,
    )
    result = sample_1(run_config)

    assert result.stop_reason != "deadline", (
        "stop_reason should NOT be 'deadline' when timesteps budget (not deadline) caused early termination. Got 'deadline' but expected something else."
    )


def test_post_process_stdout(capsys):
    mock_log = MagicMock()
    mock_log.__iter__ = MagicMock(return_value=iter(["LOG1", "LOG2"]))

    mock_trace = MagicMock()
    mock_trace.__iter__ = MagicMock(return_value=iter(["TRACE1", "TRACE2"]))

    post_process_stdout(mock_log, mock_trace)

    captured = capsys.readouterr()
    assert "LOG1" in captured.out
    assert "LOG2" in captured.out
    assert "TRACE1" in captured.out
    assert "TRACE2" in captured.out


@patch("experiments.experiment_sampler.data_writer")
@patch("experiments.experiment_sampler.ensure_parent")
def test_post_process(mock_ensure_parent, mock_data_writer):
    mock_file = MagicMock()
    mock_data_writer.return_value.__enter__ = MagicMock(return_value=mock_file)
    mock_data_writer.return_value.__exit__ = MagicMock(return_value=False)

    mock_log = MagicMock()
    mock_log.__iter__ = MagicMock(return_value=iter(["LOG1", "LOG2"]))

    mock_trace = MagicMock()
    mock_trace.__iter__ = MagicMock(return_value=iter(["TRACE1"]))

    post_process(mock_log, mock_trace, "/path/to/trace")

    mock_ensure_parent.assert_called_once_with("/path/to/trace")
    mock_data_writer.assert_called_once_with("/path/to/trace")
    assert mock_file.write.call_count == 3
    mock_file.write.assert_any_call("LOG1\n")
    mock_file.write.assert_any_call("LOG2\n")
    mock_file.write.assert_any_call("TRACE1\n")


@patch("experiments.experiment_sampler.post_process")
@patch("experiments.experiment_sampler.sample_1")
def test_scan_local(mock_sample_1, mock_post_process):
    mock_collector_log = MagicMock()
    mock_collector_trace = MagicMock()
    mock_trace_records = MagicMock()
    mock_sample_1.return_value = _SampleResult(
        collector_log=mock_collector_log,
        collector_trace=mock_collector_trace,
        trace_records=mock_trace_records,
        stop_reason="completed",
    )

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
            num_rounds=5,
            num_arms=3,
            num_denoise=10,
            num_denoise_passive=None,
            max_proposal_seconds=50.0,
            b_trace=True,
            trace_fn="/path/b",
        ),
    ]

    scan_local(run_configs)

    assert mock_sample_1.call_count == 2
    assert mock_post_process.call_count == 2


@patch("experiments.experiment_sampler.post_process")
@patch("experiments.experiment_sampler.sample_1")
def test_scan_local_single_replicate_stays_in_process(mock_sample_1, mock_post_process):
    mock_sample_1.return_value = _SampleResult(
        collector_log=MagicMock(),
        collector_trace=MagicMock(),
        trace_records=MagicMock(),
        stop_reason="completed",
    )
    mock_problem = _make_mock_problem()
    mock_problem.env.env_name = "f:sphere-2d"
    run_config = RunConfig(
        problem=mock_problem,
        opt_name="ucb",
        num_rounds=1,
        num_arms=1,
        num_denoise=None,
        num_denoise_passive=None,
        max_proposal_seconds=None,
        b_trace=True,
        trace_fn="/path/one",
    )

    scan_local([run_config], local_workers=8)

    mock_sample_1.assert_called_once()
    mock_post_process.assert_called_once()


def test_scan_local_parallel_closes_pool_on_success(monkeypatch):
    state = {"close": 0, "terminate": 0, "join": 0}

    def imap_unordered(self, _fn, items):
        for _ in items:
            yield "ok"

    def close(self):
        state["close"] += 1

    def terminate(self):
        state["terminate"] += 1

    def join(self):
        state["join"] += 1

    _FakePool = type(
        "_FakePool",
        (),
        {
            "imap_unordered": imap_unordered,
            "close": close,
            "terminate": terminate,
            "join": join,
        },
    )

    def pool(self, processes, initializer):
        assert int(processes) == 2
        assert initializer is not None
        return _FakePool()

    _FakeContext = type("_FakeContext", (), {"Pool": pool})

    monkeypatch.setattr(
        "experiments.experiment_sampler.mp.get_context",
        lambda *args, **kwargs: _FakeContext(),
    )
    _scan_local_parallel([MagicMock(), MagicMock()], max_workers=2)

    assert state["close"] == 1
    assert state["terminate"] == 0
    assert state["join"] == 1


def test_scan_local_parallel_terminates_pool_on_keyboard_interrupt(monkeypatch):
    state = {"close": 0, "terminate": 0, "join": 0}

    def imap_unordered(self, _fn, _items):
        raise KeyboardInterrupt

    def close(self):
        state["close"] += 1

    def terminate(self):
        state["terminate"] += 1

    def join(self):
        state["join"] += 1

    _FakePool = type(
        "_FakePool",
        (),
        {
            "imap_unordered": imap_unordered,
            "close": close,
            "terminate": terminate,
            "join": join,
        },
    )

    def pool(self, processes, initializer):
        assert int(processes) == 2
        assert initializer is not None
        return _FakePool()

    _FakeContext = type("_FakeContext", (), {"Pool": pool})

    monkeypatch.setattr(
        "experiments.experiment_sampler.mp.get_context",
        lambda *args, **kwargs: _FakeContext(),
    )
    with pytest.raises(KeyboardInterrupt):
        _scan_local_parallel([MagicMock(), MagicMock()], max_workers=2)

    assert state["close"] == 0
    assert state["terminate"] == 1
    assert state["join"] == 1


@patch("experiments.experiment_sampler.mk_replicates")
def test_sampler(mock_mk_replicates):
    mock_run_config = MagicMock(spec=RunConfig)
    mock_mk_replicates.return_value = [mock_run_config]

    mock_distributor = MagicMock()
    config = ExperimentConfig(
        exp_dir="/results",
        env_tag="f:ackley-10d",
        opt_name="ucb",
        num_arms=5,
        num_rounds=10,
        num_reps=1,
        policy_tag="pure-function",
    )

    sampler(config, mock_distributor)

    mock_mk_replicates.assert_called_once_with(config)
    mock_distributor.assert_called_once_with([mock_run_config])


def test_experiment_config_from_dict_missing_policy_tag_raises():
    """ExperimentConfig.from_dict must raise ValueError when policy_tag is missing."""
    d = {
        "exp_dir": "/results/exp1",
        "env_tag": "f:ackley-10d",
        "opt_name": "ucb",
        "num_arms": 5,
        "num_rounds": 10,
        "num_reps": 3,
    }
    with pytest.raises(ValueError, match="Missing required field 'policy_tag'"):
        ExperimentConfig.from_dict(d)
