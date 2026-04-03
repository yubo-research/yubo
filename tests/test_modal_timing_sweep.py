"""Tests for experiments/modal_timing_sweep.py."""

from types import SimpleNamespace
from unittest.mock import patch

import experiments.modal_timing_sweep as mts
from experiments.experiment_sampler import TIMING_SWEEP_MAX_CUMULATIVE_PROPOSAL_SECONDS
from experiments.modal_timing_sweep import clean_up, collect, prep_timing_sweep, run_timing_sweep, status


def test_prep_timing_sweep():
    sweep_tuples = [
        ("random", "tlunar:fn", 50, 30, 50, None),
        ("optuna", "push:fn", 10, 100, 10, 30),
    ]
    configs = prep_timing_sweep("results", sweep_tuples, exp_dir="exp_test")

    assert len(configs) == 2
    assert configs[0].opt_name == "random"
    assert configs[0].env_tag == "tlunar:fn"
    assert configs[0].num_arms == 50
    assert configs[0].num_rounds == 30
    assert configs[0].num_denoise == 50
    assert configs[0].num_denoise_passive is None
    assert configs[0].num_reps == 1
    assert configs[0].max_total_seconds is None
    assert configs[0].max_proposal_seconds == TIMING_SWEEP_MAX_CUMULATIVE_PROPOSAL_SECONDS

    assert configs[1].opt_name == "optuna"
    assert configs[1].env_tag == "push:fn"
    assert configs[1].num_denoise_passive == 30
    assert configs[1].max_total_seconds is None
    assert configs[1].max_proposal_seconds == TIMING_SWEEP_MAX_CUMULATIVE_PROPOSAL_SECONDS


def test_timing_sweep_worker(monkeypatch):
    import experiments.modal_timing_sweep as mts

    class _FakeDict(dict):
        def len(self):
            return len(self)

    res_dict = _FakeDict()
    monkeypatch.setattr(mts, "_results_dict", lambda: res_dict)
    monkeypatch.setattr(
        mts,
        "sample_1",
        lambda run_cfg: SimpleNamespace(
            collector_log="log",
            collector_trace="trace",
            trace_records=[{"x": 1}],
            stop_reason="completed",
        ),
    )

    run_config = SimpleNamespace(trace_fn="trace0")
    mts.timing_sweep_worker.get_raw_f()(("k0", run_config))

    assert "k0" in res_dict
    result = res_dict["k0"]
    assert result[0] == "trace0"
    assert result[1] == "log"
    assert result[2] == "trace"
    assert result[3] == [{"x": 1}]
    assert isinstance(result[4], float)
    assert result[5] == "completed"


def test_timing_sweep_resubmitter(monkeypatch):
    import experiments.modal_timing_sweep as mts

    class _FakeDict(dict):
        def len(self):
            return len(self)

    submitted = _FakeDict()
    monkeypatch.setattr(mts, "_submitted_dict", lambda: submitted)

    spawned = []

    class _Func:
        def spawn_map(self, todo):
            spawned.extend(list(todo))

    monkeypatch.setattr(mts.modal.Function, "from_name", lambda app_name, name: _Func())

    run_config = SimpleNamespace(trace_fn="t1")
    mts.timing_sweep_resubmitter.get_raw_f()([("k1", run_config, False)])

    assert submitted["k1"] is True
    assert len(spawned) == 1


def test_submit_configs(monkeypatch):
    import experiments.modal_timing_sweep as mts

    class _FakeDict(dict):
        def len(self):
            return len(self)

    monkeypatch.setattr(mts, "_results_dict", lambda: _FakeDict())
    monkeypatch.setattr(mts, "_submitted_dict", lambda: _FakeDict())
    monkeypatch.setattr(mts, "data_is_done", lambda trace_fn: False)

    spawned = []

    class _Func:
        def spawn(self, payload):
            spawned.append(payload)

    monkeypatch.setattr(mts.modal.Function, "from_name", lambda app_name, name: _Func())

    with patch("experiments.modal_timing_sweep.mk_replicates") as mock_mk:
        mock_mk.return_value = [SimpleNamespace(trace_fn="t1", deadline=None)]

        from experiments.experiment_sampler import ExperimentConfig

        config = ExperimentConfig(
            exp_dir="results/exp_test",
            env_tag="tlunar:fn",
            opt_name="random",
            num_arms=50,
            num_rounds=30,
            num_reps=1,
            num_denoise=50,
        )
        mts.submit_configs("test_batch", [config], force=False)

    assert len(spawned) == 1


def test_timing_sweep_worker_clears_wall_deadline(monkeypatch):
    """Timing sweep stops on cumulative proposal time, not wall-clock deadline."""

    import experiments.modal_timing_sweep as mts

    class _FakeDict(dict):
        def len(self):
            return len(self)

    res_dict = _FakeDict()
    monkeypatch.setattr(mts, "_results_dict", lambda: res_dict)

    captured_deadline = []

    def fake_sample_1(run_config):
        captured_deadline.append(run_config.deadline)
        return SimpleNamespace(
            collector_log="log",
            collector_trace="trace",
            trace_records=[],
            stop_reason="completed",
        )

    monkeypatch.setattr(mts, "sample_1", fake_sample_1)

    run_config = SimpleNamespace(trace_fn="trace0", deadline=1e12)
    mts.timing_sweep_worker.get_raw_f()(("k0", run_config))

    assert len(captured_deadline) == 1
    assert captured_deadline[0] is None


def test_timing_sweep_deleter(monkeypatch):
    import experiments.modal_timing_sweep as mts

    class _FakeDict(dict):
        def len(self):
            return len(self)

    res_dict = _FakeDict({"k1": "val1", "k2": "val2"})
    monkeypatch.setattr(mts, "_results_dict", lambda: res_dict)

    mts.timing_sweep_deleter.get_raw_f()(["k1"])
    assert "k1" not in res_dict
    assert "k2" in res_dict


def test_collect(monkeypatch):
    class _FakeDict(dict):
        def len(self):
            return len(self)

    res_dict = _FakeDict()
    res_dict["k1"] = ("trace_fn", "log", "trace", [{"x": 1}], 10.5, "completed")
    res_dict["k2"] = ("trace_fn2", "log2", "trace2", None)

    monkeypatch.setattr(mts, "_results_dict", lambda: res_dict)
    monkeypatch.setattr(mts, "data_is_done", lambda trace_fn: False)
    monkeypatch.setattr(mts, "post_process", lambda *args, **kwargs: None)

    spawned = []

    class _Func:
        def spawn(self, payload):
            spawned.append(payload)

    monkeypatch.setattr(mts.modal.Function, "from_name", lambda app_name, name: _Func())

    collect()

    assert len(spawned) == 1
    assert "k1" in spawned[0]
    assert "k2" in spawned[0]


def test_status(monkeypatch, capsys):
    class _FakeDict(dict):
        def len(self):
            return len(self)

    monkeypatch.setattr(mts, "_results_dict", lambda: _FakeDict({"a": 1}))
    monkeypatch.setattr(mts, "_submitted_dict", lambda: _FakeDict({"b": 1, "c": 1}))

    status()

    captured = capsys.readouterr()
    assert "results_available = 1" in captured.out
    assert "submitted = 2" in captured.out


def test_clean_up(monkeypatch, capsys):
    deleted = []
    monkeypatch.setattr(mts.modal.Dict, "delete", lambda name: deleted.append(name))

    clean_up()

    assert "timing_sweep_dict" in deleted
    assert "timing_sweep_submitted_dict" in deleted


def test_run_timing_sweep(monkeypatch):
    submitted = []

    def fake_submit(batch_tag, configs, force):
        submitted.append((batch_tag, configs, force))

    monkeypatch.setattr(mts, "submit_configs", fake_submit)

    sweep_tuples = [("random", "tlunar:fn", 50, 30, 50, None)]
    run_timing_sweep("test_batch", sweep_tuples)

    assert len(submitted) == 1
    assert submitted[0][0] == "test_batch"
    assert len(submitted[0][1]) == 1
