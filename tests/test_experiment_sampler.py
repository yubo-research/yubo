import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from experiments.experiment_sampler import (
    ExperimentConfig,
    RunConfig,
    _resolve_video_dir,
    extract_trace_fns,
    mk_replicates,
    post_process,
    post_process_stdout,
    prep_args_1,
    prep_d_args,
    sample_1,
    sampler,
    scan_local,
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


def test_extract_trace_fns():
    mock_env = MagicMock()
    run_configs = [
        RunConfig(
            env_conf=mock_env,
            env_tag="f:ackley-10d",
            problem_seed=0,
            noise_seed_0=0,
            opt_name="ucb",
            num_rounds=10,
            num_arms=5,
            num_denoise=None,
            num_denoise_passive=None,
            max_proposal_seconds=None,
            rollout_workers=None,
            b_trace=True,
            trace_fn="/path/a",
        ),
        RunConfig(
            env_conf=mock_env,
            env_tag="f:ackley-10d",
            problem_seed=1,
            noise_seed_0=10,
            opt_name="ei",
            num_rounds=10,
            num_arms=5,
            num_denoise=None,
            num_denoise_passive=None,
            max_proposal_seconds=None,
            rollout_workers=None,
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
    )
    assert isinstance(result, ExperimentConfig)
    assert result.exp_dir == "/results/exp1"
    assert result.env_tag == "f:ackley-10d"
    assert result.opt_name == "ucb"
    assert result.num_arms == 5
    assert result.num_reps == 3
    assert result.num_rounds == 10
    assert result.num_denoise == 100


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
    )
    assert len(results) == 2 * 2 * 2 * 1
    assert all(isinstance(r, ExperimentConfig) for r in results)
    assert results[0].env_tag == "f:ackley-5d"
    assert results[0].opt_name == "ucb"


@patch("experiments.experiment_sampler.data_is_done")
def test_mk_replicates(mock_data_is_done):
    mock_data_is_done.return_value = False

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
        )
        results = mk_replicates(config)

    assert len(results) == 3
    assert all(isinstance(r, RunConfig) for r in results)
    assert results[0].opt_name == "ucb"
    assert results[0].num_arms == 5
    assert results[0].num_rounds == 10
    assert results[0].num_denoise == 100
    assert results[0].b_trace is True
    assert results[0].env_conf is None
    assert results[0].problem_seed == 18
    assert results[0].noise_seed_0 == 180


@patch("experiments.experiment_sampler.data_is_done")
def test_mk_replicates_skips_done(mock_data_is_done):
    mock_data_is_done.return_value = True

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExperimentConfig(
            exp_dir=tmpdir,
            env_tag="f:ackley-10d",
            opt_name="ucb",
            num_arms=5,
            num_rounds=10,
            num_reps=3,
            num_denoise=None,
        )
        results = mk_replicates(config)

    assert len(results) == 0


def test_mk_replicates_creates_out_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("experiments.experiment_sampler.data_is_done") as mock_data_is_done:
            mock_data_is_done.return_value = False

            config = ExperimentConfig(
                exp_dir=tmpdir,
                env_tag="f:test-5d",
                opt_name="random",
                num_arms=2,
                num_rounds=5,
                num_reps=1,
                num_denoise=None,
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
    )
    d = config.to_dict()
    assert d["exp_dir"] == "/results/exp1"
    assert d["env_tag"] == "f:ackley-10d"
    assert d["opt_name"] == "ucb"
    assert d["num_arms"] == 5
    assert d["num_rounds"] == 10
    assert d["num_reps"] == 3
    assert d["num_denoise"] == 100


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


def test_experiment_config_from_dict_none_denoise():
    d = {
        "exp_dir": "/results/exp1",
        "env_tag": "f:ackley-10d",
        "opt_name": "ucb",
        "num_arms": 5,
        "num_rounds": 10,
        "num_reps": 3,
        "num_denoise": "None",
    }
    config = ExperimentConfig.from_dict(d)
    assert config.num_denoise is None


def test_sampler_forwards_max_total_seconds_to_wrapped_distributor(monkeypatch):
    config = ExperimentConfig(
        exp_dir="/results/exp1",
        env_tag="f:ackley-10d",
        opt_name="ucb",
        num_arms=5,
        num_rounds=10,
        num_reps=1,
        max_total_seconds=12.5,
    )
    monkeypatch.setattr("experiments.experiment_sampler.mk_replicates", lambda _config: [MagicMock()])

    calls = {}

    def wrapped_distributor(run_configs, max_total_seconds=None):
        calls["num_runs"] = len(run_configs)
        calls["max_total_seconds"] = max_total_seconds

    sampler(config, wrapped_distributor)
    assert calls["num_runs"] == 1
    assert calls["max_total_seconds"] == 12.5


def test_sampler_handles_distributor_without_timeout_param(monkeypatch):
    config = ExperimentConfig(
        exp_dir="/results/exp1",
        env_tag="f:ackley-10d",
        opt_name="ucb",
        num_arms=5,
        num_rounds=10,
        num_reps=1,
        max_total_seconds=12.5,
    )
    monkeypatch.setattr("experiments.experiment_sampler.mk_replicates", lambda _config: [MagicMock(), MagicMock()])

    calls = {}

    def distributor_without_timeout(run_configs):
        calls["num_runs"] = len(run_configs)

    sampler(config, distributor_without_timeout)
    assert calls["num_runs"] == 2


def test_sampler_fallback_when_signature_introspection_fails(monkeypatch):
    config = ExperimentConfig(
        exp_dir="/results/exp1",
        env_tag="f:ackley-10d",
        opt_name="ucb",
        num_arms=5,
        num_rounds=10,
        num_reps=1,
        max_total_seconds=12.5,
    )
    monkeypatch.setattr("experiments.experiment_sampler.mk_replicates", lambda _config: [MagicMock()])
    monkeypatch.setattr("experiments.experiment_sampler.inspect.signature", lambda _fn: (_ for _ in ()).throw(TypeError()))

    calls = {}

    def distributor_without_timeout(run_configs):
        calls["num_runs"] = len(run_configs)

    sampler(config, distributor_without_timeout)
    assert calls["num_runs"] == 1


def test_scan_parallel_empty_run_configs(monkeypatch, capsys):
    import concurrent.futures as cf

    from experiments.experiment_sampler import scan_parallel

    class DummyFuture:
        def result(self):
            return None

    class DummyExecutor:
        def __init__(self, *args, **kwargs):
            _ = args
            self._max_workers = kwargs.get("max_workers")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = (exc_type, exc, tb)
            return False

        def submit(self, _fn, _rc):
            return DummyFuture()

    monkeypatch.setattr(cf, "ProcessPoolExecutor", DummyExecutor)
    monkeypatch.setattr(cf, "as_completed", lambda _futs: [])

    scan_parallel([], max_total_seconds=None, max_workers=2)
    out = capsys.readouterr().out
    assert "TIME_PARALLEL:" in out


@patch("experiments.experiment_sampler.Optimizer")
@patch("experiments.experiment_sampler.default_policy")
@patch("experiments.experiment_sampler.seed_all")
@patch("experiments.experiment_sampler.torch")
def test_sample_1(mock_torch, mock_seed_all, mock_default_policy, mock_optimizer_class):
    mock_torch.cuda.is_available.return_value = False
    mock_torch.empty.return_value.device = "cpu"

    mock_policy = MagicMock()
    mock_default_policy.return_value = mock_policy

    mock_trace_entry = MagicMock()
    mock_trace_entry.dt_prop = 0.1
    mock_trace_entry.dt_eval = 0.2
    mock_trace_entry.rreturn = 1.5

    mock_optimizer = MagicMock()
    mock_optimizer.collect_trace.return_value = iter([mock_trace_entry, mock_trace_entry])
    mock_optimizer_class.return_value = mock_optimizer

    mock_env_conf = MagicMock()
    mock_env_conf.problem_seed = 42
    mock_env_conf.env_name = "test_env"

    run_config = RunConfig(
        env_conf=mock_env_conf,
        env_tag="f:test-5d",
        problem_seed=0,
        noise_seed_0=0,
        opt_name="ucb",
        num_rounds=2,
        num_arms=5,
        num_denoise=None,
        num_denoise_passive=None,
        max_proposal_seconds=100.0,
        rollout_workers=None,
        b_trace=True,
        trace_fn="/path/to/trace",
    )
    collector_log, collector_trace, trace_records = sample_1(run_config)

    mock_seed_all.assert_called_once_with(42 + 27)
    mock_torch.set_default_device.assert_not_called()
    mock_default_policy.assert_called_once_with(mock_env_conf)
    mock_optimizer_class.assert_called_once()
    mock_optimizer.collect_trace.assert_called_once_with(
        designer_name="ucb",
        max_iterations=2,
        max_proposal_seconds=100.0,
        deadline=None,
        resume_state=None,
        checkpoint_every=None,
        checkpoint_path=None,
    )

    trace_lines = list(collector_trace)
    assert len(trace_lines) == 3
    assert "TRACE:" in trace_lines[0]
    assert "DONE" in trace_lines[2]

    assert len(trace_records) == 2
    assert trace_records[0].i_iter == 0
    assert trace_records[0].rreturn == 1.5


@patch("experiments.experiment_sampler.get_env_conf")
@patch("experiments.experiment_sampler.Optimizer")
@patch("experiments.experiment_sampler.default_policy")
@patch("experiments.experiment_sampler.seed_all")
@patch("experiments.experiment_sampler.torch")
def test_sample_1_rebuilds_env_conf(
    mock_torch,
    mock_seed_all,
    mock_default_policy,
    mock_optimizer_class,
    mock_get_env_conf,
):
    mock_torch.cuda.is_available.return_value = False
    mock_torch.empty.return_value.device = "cpu"
    mock_default_policy.return_value = MagicMock()

    mock_trace_entry = MagicMock()
    mock_trace_entry.dt_prop = 0.1
    mock_trace_entry.dt_eval = 0.2
    mock_trace_entry.rreturn = 1.5

    mock_optimizer = MagicMock()
    mock_optimizer.collect_trace.return_value = iter([mock_trace_entry])
    mock_optimizer_class.return_value = mock_optimizer

    mock_env_conf = MagicMock()
    mock_env_conf.problem_seed = 7
    mock_env_conf.env_name = "test_env"
    mock_get_env_conf.return_value = mock_env_conf

    run_config = RunConfig(
        env_conf=None,
        env_tag="f:test-5d",
        problem_seed=7,
        noise_seed_0=70,
        opt_name="ucb",
        num_rounds=1,
        num_arms=5,
        num_denoise=None,
        num_denoise_passive=None,
        max_proposal_seconds=None,
        rollout_workers=None,
        b_trace=True,
        trace_fn="/path/to/trace",
    )
    sample_1(run_config)

    mock_get_env_conf.assert_called_once_with(
        "f:test-5d",
        problem_seed=7,
        noise_level=None,
        noise_seed_0=70,
    )
    mock_seed_all.assert_called_once_with(7 + 27)
    mock_torch.set_default_device.assert_not_called()


@patch("experiments.experiment_sampler.Optimizer")
@patch("experiments.experiment_sampler.default_policy")
@patch("experiments.experiment_sampler.seed_all")
@patch("experiments.experiment_sampler.torch")
def test_sample_1_no_trace(mock_torch, mock_seed_all, mock_default_policy, mock_optimizer_class):
    mock_torch.cuda.is_available.return_value = False
    mock_torch.empty.return_value.device = "cpu"
    mock_default_policy.return_value = MagicMock()

    mock_trace_entry = MagicMock()
    mock_trace_entry.dt_prop = 0.1
    mock_trace_entry.dt_eval = 0.2
    mock_trace_entry.rreturn = 1.5

    mock_optimizer = MagicMock()
    mock_optimizer.collect_trace.return_value = iter([mock_trace_entry])
    mock_optimizer_class.return_value = mock_optimizer

    mock_env_conf = MagicMock()
    mock_env_conf.problem_seed = 0

    run_config = RunConfig(
        env_conf=mock_env_conf,
        env_tag="f:test-5d",
        problem_seed=0,
        noise_seed_0=0,
        opt_name="random",
        num_rounds=1,
        num_arms=1,
        num_denoise=None,
        num_denoise_passive=None,
        max_proposal_seconds=None,
        rollout_workers=None,
        b_trace=False,
        trace_fn="/path/to/trace",
    )
    collector_log, collector_trace, trace_records = sample_1(run_config)
    mock_torch.set_default_device.assert_not_called()

    trace_lines = list(collector_trace)
    assert len(trace_lines) == 1
    assert trace_lines[0] == "DONE"

    assert len(trace_records) == 1


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
    mock_sample_1.return_value = (
        mock_collector_log,
        mock_collector_trace,
        mock_trace_records,
    )

    mock_env = MagicMock()
    run_configs = [
        RunConfig(
            env_conf=mock_env,
            env_tag="f:ackley-10d",
            problem_seed=0,
            noise_seed_0=0,
            opt_name="ucb",
            num_rounds=10,
            num_arms=5,
            num_denoise=None,
            num_denoise_passive=None,
            max_proposal_seconds=None,
            rollout_workers=None,
            b_trace=True,
            trace_fn="/path/a",
        ),
        RunConfig(
            env_conf=mock_env,
            env_tag="f:ackley-10d",
            problem_seed=1,
            noise_seed_0=10,
            opt_name="ei",
            num_rounds=5,
            num_arms=3,
            num_denoise=10,
            num_denoise_passive=None,
            max_proposal_seconds=50.0,
            rollout_workers=None,
            b_trace=True,
            trace_fn="/path/b",
        ),
    ]

    scan_local(run_configs)

    assert mock_sample_1.call_count == 2
    assert mock_post_process.call_count == 2
    mock_post_process.assert_any_call(mock_collector_log, mock_collector_trace, "/path/a", mock_trace_records)
    mock_post_process.assert_any_call(mock_collector_log, mock_collector_trace, "/path/b", mock_trace_records)


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
    )

    sampler(config, mock_distributor)

    mock_mk_replicates.assert_called_once_with(config)
    mock_distributor.assert_called_once_with([mock_run_config], max_total_seconds=None)


def test_resolve_video_dir_default_under_run_root(tmp_path):
    trace_dir = tmp_path / "a1b2c3d4" / "traces"
    trace_dir.mkdir(parents=True)
    trace_path = trace_dir / "00000"
    trace_path.write_text("")
    run_config = RunConfig(
        env_conf=None,
        env_tag="f:ackley-10d",
        problem_seed=0,
        noise_seed_0=0,
        opt_name="ucb",
        num_rounds=1,
        num_arms=1,
        num_denoise=None,
        num_denoise_passive=None,
        max_proposal_seconds=None,
        rollout_workers=None,
        b_trace=False,
        trace_fn=str(trace_path),
        video_enable=True,
        video_dir=None,
    )
    video_dir, resolved_trace = _resolve_video_dir(run_config)
    assert resolved_trace == trace_path.resolve()
    assert video_dir == (tmp_path / "a1b2c3d4" / "videos" / "00000").resolve()
    assert video_dir.is_dir()


def test_resolve_video_dir_relative_uses_cwd(tmp_path, monkeypatch):
    trace_dir = tmp_path / "a1b2c3d4" / "traces"
    trace_dir.mkdir(parents=True)
    trace_path = trace_dir / "00000"
    trace_path.write_text("")
    monkeypatch.chdir(tmp_path)
    run_config = RunConfig(
        env_conf=None,
        env_tag="f:ackley-10d",
        problem_seed=0,
        noise_seed_0=0,
        opt_name="ucb",
        num_rounds=1,
        num_arms=1,
        num_denoise=None,
        num_denoise_passive=None,
        max_proposal_seconds=None,
        rollout_workers=None,
        b_trace=False,
        trace_fn=str(trace_path),
        video_enable=True,
        video_dir="_tmp/custom_videos",
    )
    video_dir, _ = _resolve_video_dir(run_config)
    assert video_dir == (tmp_path / "_tmp" / "custom_videos").resolve()
    assert video_dir.is_dir()
