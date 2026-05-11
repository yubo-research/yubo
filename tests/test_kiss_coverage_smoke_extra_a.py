from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

from acq.acq_bt import AcqBT
from acq.acq_dpp import AcqDPP
from acq.fit_gp import _EmptyTransform
from analysis.data_locator import DataLocator


def test_kiss_cov_acqbt_x_max(monkeypatch):
    def _gp_call(self, _x):
        return SimpleNamespace(mean=torch.tensor(0.0))

    _GP = type("_GP", (), {"__call__": _gp_call})
    monkeypatch.setattr("acq.acq_bt.fit_gp.fit_gp_XY", lambda X, Y, model_spec: _GP())
    monkeypatch.setattr(
        "acq.acq_bt.find_max",
        lambda gp, bounds: torch.ones((1, bounds.shape[1]), dtype=bounds.dtype),
    )

    acq = AcqBT(
        acq_factory=lambda gp, **kwargs: SimpleNamespace(),
        data=[],
        num_dim=3,
        acq_kwargs=None,
        device=torch.device("cpu"),
        dtype=torch.double,
        num_keep=None,
        keep_style=None,
        model_spec=None,
    )
    x = acq.x_max()
    assert tuple(x.shape) == (1, 3)


def test_kiss_cov_acq_dpp_and_fitgp_empty_transform():
    def _model_init(self):
        self.train_inputs = (torch.zeros(2, 3, dtype=torch.double),)
        self.likelihood = SimpleNamespace(noise=torch.tensor(1.0, dtype=torch.double))

    def _model_eval(self):
        return None

    _Model = type("_Model", (), {"__init__": _model_init, "eval": _model_eval})

    acq = AcqDPP(_Model(), num_X_samples=8, num_runs=1)
    assert acq._num_dim == 3

    t = _EmptyTransform()
    y = torch.tensor([[1.0]])
    y2, yvar2 = t.forward(y)
    assert torch.equal(y2, y)
    assert yvar2 is None


def test_kiss_cov_data_locator_optimizers(tmp_path):
    results_dir = tmp_path / "results"
    exp_dir = results_dir / "exp_a"
    exp_dir.mkdir(parents=True)
    (exp_dir / "config.json").write_text(json.dumps({"opt_name": "random", "env_tag": "f:ackley-2d"}))
    dl = DataLocator(results_path=str(results_dir), exp_dir="", opt_names=["random", "sobol"])
    assert dl.optimizers() == ["random"]


def test_kiss_cov_ops_catalog_and_ops_data(tmp_path):
    from tests.kiss_ops_catalog_data_shared import run_kiss_ops_catalog_and_data_cli

    run_kiss_ops_catalog_and_data_cli(tmp_path)


def test_kiss_cov_modal_batches_collect_and_cleanup(monkeypatch):
    import experiments.modal_batches_impl as mb

    _FakeDict = type("_FakeDict", (dict,), {"len": lambda self: len(self)})
    res_dict = _FakeDict()
    submitted = _FakeDict()
    monkeypatch.setattr(mb, "_results_dict", lambda tag: res_dict)
    monkeypatch.setattr(mb, "_submitted_dict", lambda tag: submitted)
    monkeypatch.setattr(
        mb,
        "sample_1",
        lambda run_cfg: SimpleNamespace(
            collector_log="log",
            collector_trace="trace",
            trace_records=[{"x": 1}],
            stop_reason="completed",
        ),
    )

    def _func_spawn_map(self, _todo):
        return None

    def _func_spawn(self, *_args):
        return None

    _Func = type("_Func", (), {"spawn_map": _func_spawn_map, "spawn": _func_spawn})
    monkeypatch.setattr(mb.modal.Function, "from_name", lambda app_name, name: _Func())
    monkeypatch.setattr(mb, "_gen_jobs", lambda tag: [("k1", SimpleNamespace(trace_fn="t1"))])
    monkeypatch.setattr(mb, "data_is_done", lambda trace_fn: False)
    monkeypatch.setattr(mb, "post_process", lambda *args, **kwargs: None)

    tag = "tag"
    mb.modal_batches_worker.get_raw_f()((tag, "k0", SimpleNamespace(trace_fn="trace0")))
    mb.modal_batches_resubmitter.get_raw_f()([("k1", SimpleNamespace(trace_fn="t1"), False)], tag)
    mb.batches_submitter(tag, "batch_tag")

    res_dict["k2"] = ("trace_fn", "log", "trace", None)
    mb._collect(tag)
    mb.status(tag)
    mb.modal_batch_deleter.get_raw_f()(["k2"], tag)

    monkeypatch.setattr(mb.modal.Dict, "delete", lambda name: None)
    mb.clean_up(tag)


def test_kiss_cov_modal_collect_and_modal_learn(monkeypatch):
    import experiments.experiment_sampler as experiment_sampler

    sys.modules["experiment_sampler"] = experiment_sampler
    import experiments.modal_collect as modal_collect
    import experiments.modal_learn as modal_learn

    monkeypatch.setattr(modal_learn, "PERSISTED_KEY_COUNT", 3)

    def _call_get(self, timeout):
        assert timeout == 5
        return ("trace", "log", "collector")

    _Call = type("_Call", (), {"get": _call_get})
    _Factory = type("_Factory", (), {"from_id": staticmethod(lambda _call_id: _Call())})
    monkeypatch.setattr(modal_collect.modal.functions, "FunctionCall", _Factory)
    out = modal_collect.get_job_result("id")
    assert isinstance(out, tuple)

    called = {"collect": 0}
    monkeypatch.setattr(modal_collect, "collect", lambda job_fn, cb: cb(("trace", "log", "collector")))
    monkeypatch.setattr(
        modal_collect,
        "post_process",
        lambda *args: called.__setitem__("collect", called["collect"] + 1),
    )
    monkeypatch.setattr(modal_collect.os.path, "exists", lambda p: False)
    modal_collect.main("jobs.txt")
    assert called["collect"] == 1

    def _queue_init(self):
        self._n = 0

    def _queue_put(self, _x):
        return None

    def _queue_get(self, block=True, timeout=10):
        _ = (block, timeout)
        if self._n == 0:
            self._n += 1
            return "k0"
        raise modal_learn.queue.Empty()

    _Queue = type("_Queue", (), {"__init__": _queue_init, "put": _queue_put, "get": _queue_get})

    def _dict_getitem(self, key):
        return dict.get(self, key, ("missing", "missing", 0.0))

    _Dict = type("_Dict", (dict,), {"__getitem__": _dict_getitem})
    monkeypatch.setattr(
        modal_learn.modal.Queue,
        "from_name",
        lambda name, create_if_missing=True: _Queue(),
    )
    monkeypatch.setattr(
        modal_learn.modal.Dict,
        "from_name",
        lambda name, create_if_missing=True: _Dict({"key_0": ("a", "b", 1.0)}),
    )
    modal_learn.process_job.get_raw_f()("processor")
    modal_learn.get_job_result()
    monkeypatch.setattr(modal_learn, "start", lambda cmd: None)
    modal_learn.main("start")
    modal_learn.main("submit")
    modal_learn.main("get")


def test_kiss_cov_modal_image_and_interactive(monkeypatch):
    import experiments.experiment_sampler as experiment_sampler
    import experiments.modal_image as modal_image

    sys.modules["experiment_sampler"] = experiment_sampler
    import experiments.modal_interactive as modal_interactive

    img = modal_image.mk_image()
    assert img is not None

    rc = SimpleNamespace(trace_fn="trace.jsonl")
    sample_1_return = SimpleNamespace(
        collector_log=("log",),
        collector_trace=("trace",),
        trace_records=[],
        stop_reason=None,
    )
    modal_remote_tuple = (
        sample_1_return.collector_log,
        sample_1_return.collector_trace,
        sample_1_return.trace_records,
        sample_1_return.stop_reason,
    )

    def fake_sample_1(run_config):
        assert run_config is rc
        return sample_1_return

    monkeypatch.setattr(modal_interactive, "sample_1", fake_sample_1)
    out = modal_interactive.modal_sample_1.get_raw_f()(rc)
    assert isinstance(out, tuple)
    assert out == modal_remote_tuple
    monkeypatch.setattr(modal_interactive, "mk_replicates", lambda config: [rc])
    monkeypatch.setattr(modal_interactive, "post_process", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        modal_interactive.modal_sample_1,
        "remote",
        lambda _run_config: modal_remote_tuple,
    )
    modal_interactive.run_job("exp", "env", "opt", 1, 1, 1)


def test_kiss_cov_fig_utils(monkeypatch, tmp_path):
    from figures.mtv import fig_util

    fake_env = SimpleNamespace()
    fake_problem = SimpleNamespace(env=fake_env, build_policy=lambda: "policy")
    monkeypatch.setattr(fig_util, "build_problem", lambda *a, **k: fake_problem)
    ep = fig_util.expository_problem()
    assert ep.opt_name == "mtv"
    assert isinstance(fig_util.show(torch.tensor([1.0, 2.0])), str)
    mesh = fig_util.mk_mesh(n=4)
    fig_util.dump_mesh(str(tmp_path), "mesh.txt", mesh.x_1, mesh.x_2, np.zeros_like(mesh.x_1))

    def _fig_env_step(self, x):
        return None, float(np.sum(x)), False, False

    _FigEnv = type("_FigEnv", (), {"step": _fig_env_step})

    def _fig_envconf_make(self):
        return _FigEnv()

    _FigEnvConf = type("_FigEnvConf", (), {"make": _fig_envconf_make})

    def _post_init(self, n):
        self.mean = torch.zeros((n, 1))
        self.variance = torch.ones((n, 1))
        self._n = n

    def _post_sample(self, size):
        return torch.zeros(size + torch.Size([self._n]))

    _Post = type("_Post", (), {"__init__": _post_init, "sample": _post_sample})

    def _gp_posterior(self, xs):
        return _Post(len(xs))

    _GPfig = type("_GPfig", (), {"posterior": _gp_posterior})

    fig_util.mean_func_contours(str(tmp_path), _FigEnvConf())
    fig_util.mean_gp_contours(str(tmp_path), _GPfig())
    fig_util.var_contours(str(tmp_path), _GPfig())
    fig_util.pmax_contours(str(tmp_path), _GPfig())


def test_kiss_cov_fig_pstar_scale_and_turbo_best_datum(monkeypatch, tmp_path):
    from figures.pts import fig_pstar_scale as fps
    from optimizer.turbo_enn_designer import TurboENNDesigner

    monkeypatch.setattr(fps, "_num_dims", [2])
    d_args = fps.dist_pstar_scales_all_funcs("mtv", 2)
    assert d_args

    def _dm_init(self, app_name, fn_name, job_fn):
        _ = (app_name, fn_name, job_fn)

    def _dm_call(self, all_args):
        assert all_args

    _DM = type("_DM", (), {"__init__": _dm_init, "__call__": _dm_call})
    monkeypatch.setattr(fps, "DistModal", _DM)
    monkeypatch.setattr(fps, "dist_pstar_scales_all_funcs", lambda designer, num_dim: [{"x": 1}])
    fps.distribute("mtv", "jobs.txt", dry_run=False)

    monkeypatch.setattr(
        fps,
        "collect",
        lambda job_fn, cb: cb(("designer", 2, "f:sphere-2d", [("x", 1)])),
    )
    os.makedirs(tmp_path / "fig_data" / "sts", exist_ok=True)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        fps.collect_all("jobs.txt")
    finally:
        os.chdir(cwd)

    monkeypatch.setattr(fps, "distribute", lambda *a, **k: None)
    monkeypatch.setattr(fps, "collect_all", lambda *a, **k: None)
    fps.spawn_all("dist", "jobs.txt", False, "mtv")
    fps.spawn_all("collect", "jobs.txt", False, "mtv")

    d = TurboENNDesigner(
        policy=SimpleNamespace(num_params=lambda: 2),
        turbo_mode="turbo-zero",
    )
    assert d.best_datum() is None
