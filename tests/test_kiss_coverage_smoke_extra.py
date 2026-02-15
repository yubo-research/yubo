from __future__ import annotations

import json
import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import torch
from click.testing import CliRunner
from torch import nn

from acq.acq_bt import AcqBT
from acq.acq_dpp import AcqDPP
from acq.fit_gp import _EmptyTransform
from analysis.data_locator import DataLocator
from optimizer.bt_designer import BTDesigner
from optimizer.designer_registry import _SimpleContext
from problems.noise_maker import NoiseMaker
from problems.policy_mixin import PolicyParamsMixin


def test_kiss_cov_acqbt_x_max(monkeypatch):
    class _GP:
        def __call__(self, _x):
            return SimpleNamespace(mean=torch.tensor(0.0))

    monkeypatch.setattr("acq.acq_bt.fit_gp.fit_gp_XY", lambda X, Y, model_spec: _GP())
    monkeypatch.setattr("acq.acq_bt.find_max", lambda gp, bounds: torch.ones((1, bounds.shape[1]), dtype=bounds.dtype))

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
    class _Model:
        def __init__(self):
            self.train_inputs = (torch.zeros(2, 3, dtype=torch.double),)
            self.likelihood = SimpleNamespace(noise=torch.tensor(1.0, dtype=torch.double))

        def eval(self):
            return None

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
    import ops.catalog as catalog
    import ops.data as data_cli

    runner = CliRunner()
    catalog.cli.callback()
    catalog.environments.callback()
    res = runner.invoke(catalog.cli, ["environments"])
    assert res.exit_code == 0

    results_dir = tmp_path / "results"
    exp_dir = results_dir / "abc123"
    traces = exp_dir / "traces"
    traces.mkdir(parents=True)
    (exp_dir / "config.json").write_text(json.dumps({"opt_name": "random", "env_tag": "f:ackley-2d", "num_arms": 1, "num_rounds": 1}))
    (traces / "00000.jsonl").write_text("{}\n")

    data_cli.cli.callback()
    data_cli.ls.callback(results_dir, False)
    res = runner.invoke(data_cli.cli, ["rm", str(results_dir), "abc123", "-f"])
    assert res.exit_code == 0

    exp_dir.mkdir(parents=True)
    (exp_dir / "config.json").write_text(json.dumps({"opt_name": "random", "env_tag": "f:ackley-2d"}))
    data_cli.rm.callback(results_dir, ("abc123",), True)
    assert not exp_dir.exists()


def test_kiss_cov_designer_registry_and_best_datum():
    ctx = _SimpleContext(
        policy=object(),
        num_arms=1,
        bt=lambda *a, **k: None,
        num_keep=None,
        keep_style=None,
        num_keep_val=None,
        init_yubo_default=1,
        init_ax_default=1,
        default_num_X_samples=1,
    )
    assert ctx.num_arms == 1

    d = BTDesigner(
        policy=SimpleNamespace(),
        acq_fn=lambda *a, **k: None,
        num_restarts=1,
        raw_samples=1,
        start_at_max=False,
    )
    assert d.best_datum() is None


def test_kiss_cov_noise_maker_and_policy_mixin():
    class _DummySpace:
        shape = (1,)

    class _DummyEnv:
        observation_space = _DummySpace()
        action_space = _DummySpace()

        def step(self, _action):
            return None, 1.0, False, False, {}

        def close(self):
            return None

    noise = NoiseMaker(_DummyEnv(), normalized_noise_level=0.0, num_measurements=2)
    assert noise.observation_space is not None
    assert noise.action_space is not None

    class _Policy(nn.Module, PolicyParamsMixin):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(2, 1, bias=False)
            with torch.inference_mode():
                fp = torch.cat([p.detach().reshape(-1) for p in self.parameters()]).cpu().numpy()
            self._flat_params_init = fp
            self._const_scale = 1.0

    p = _Policy()
    assert p.num_params() == 2


def test_kiss_cov_dist_modal_collect(monkeypatch, tmp_path):
    import experiments.dist_modal as dist_modal

    class _Call:
        def get(self, timeout):
            assert timeout == 5
            return {"ok": True}

    class _Factory:
        @staticmethod
        def from_id(_call_id):
            return _Call()

    monkeypatch.setattr(dist_modal.modal.functions, "FunctionCall", _Factory)
    fn = tmp_path / "jobs.txt"
    fn.write_text("abc\n")
    got = []
    dist_modal.collect(str(fn), lambda x: got.append(x))
    assert got == [{"ok": True}]


def test_kiss_cov_modal_batches_collect_and_cleanup(monkeypatch):
    import experiments.modal_batches as mb

    class _FakeDict(dict):
        def len(self):
            return len(self)

    res_dict = _FakeDict()
    submitted = _FakeDict()
    monkeypatch.setattr(mb, "_results_dict", lambda: res_dict)
    monkeypatch.setattr(mb, "_submitted_dict", lambda: submitted)
    monkeypatch.setattr(mb, "sample_1", lambda run_cfg: ("log", "trace", [{"x": 1}]))

    class _Func:
        def spawn_map(self, _todo):
            return None

        def spawn(self, _payload):
            return None

    monkeypatch.setattr(mb.modal.Function, "from_name", lambda app_name, name: _Func())
    monkeypatch.setattr(mb, "_gen_jobs", lambda tag: [("k1", SimpleNamespace(trace_fn="t1"))])
    monkeypatch.setattr(mb, "data_is_done", lambda trace_fn: False)
    monkeypatch.setattr(mb, "post_process", lambda *args, **kwargs: None)

    mb.modal_batches_worker.get_raw_f()(("k0", SimpleNamespace(trace_fn="trace0")))
    mb.modal_batches_resubmitter.get_raw_f()([("k1", SimpleNamespace(trace_fn="t1"), False)])
    mb.batches_submitter("tag")

    res_dict["k2"] = ("trace_fn", "log", "trace", None)
    mb.collect()
    mb.status()
    mb.modal_batch_deleter.get_raw_f()(["k2"])

    monkeypatch.setattr(mb.modal.Dict, "delete", lambda name: None)
    mb.clean_up()


def test_kiss_cov_modal_collect_and_modal_learn(monkeypatch):
    import experiments.experiment_sampler as experiment_sampler

    sys.modules["experiment_sampler"] = experiment_sampler
    import experiments.modal_collect as modal_collect
    import experiments.modal_learn as modal_learn

    class _Call:
        def get(self, timeout):
            assert timeout == 5
            return ("trace", "log", "collector")

    class _Factory:
        @staticmethod
        def from_id(_call_id):
            return _Call()

    monkeypatch.setattr(modal_collect.modal.functions, "FunctionCall", _Factory)
    out = modal_collect.get_job_result("id")
    assert isinstance(out, tuple)

    called = {"collect": 0}
    monkeypatch.setattr(modal_collect, "collect", lambda job_fn, cb: cb(("trace", "log", "collector")))
    monkeypatch.setattr(modal_collect, "post_process", lambda *args: called.__setitem__("collect", called["collect"] + 1))
    monkeypatch.setattr(modal_collect.os.path, "exists", lambda p: False)
    modal_collect.main("jobs.txt")
    assert called["collect"] == 1

    class _Queue:
        def __init__(self):
            self._n = 0

        def put(self, _x):
            return None

        def get(self, block=True, timeout=10):
            _ = (block, timeout)
            if self._n == 0:
                self._n += 1
                return "k0"
            raise modal_learn.queue.Empty()

    class _Dict(dict):
        def __getitem__(self, key):
            return super().get(key, ("missing", "missing", 0.0))

    monkeypatch.setattr(modal_learn.modal.Queue, "from_name", lambda name, create_if_missing=True: _Queue())
    monkeypatch.setattr(modal_learn.modal.Dict, "from_name", lambda name, create_if_missing=True: _Dict({"key_0": ("a", "b", 1.0)}))
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

    monkeypatch.setattr(modal_interactive, "sample_1", lambda **kwargs: ("log", "trace"))
    out = modal_interactive.modal_sample_1.get_raw_f()({"a": 1})
    assert isinstance(out, tuple)
    monkeypatch.setattr(modal_interactive, "mk_replicates", lambda d: [dict(d, trace_fn="trace.jsonl")])
    monkeypatch.setattr(modal_interactive, "post_process", lambda *args: None)
    monkeypatch.setattr(modal_interactive.modal_sample_1, "remote", lambda d: ("log", "trace"))
    modal_interactive.run_job("exp", "env", "opt", 1, 1, 1)


def test_kiss_cov_compare_to_gp_and_fig_utils(monkeypatch, tmp_path):
    fake_tp = types.ModuleType("third_party")
    fake_enn_pkg = types.ModuleType("third_party.enn")
    fake_enn_subpkg = types.ModuleType("third_party.enn.enn")
    fake_enn_params = types.ModuleType("third_party.enn.enn.enn_params")
    fake_enn_params.PosteriorFlags = lambda observation_noise=False: SimpleNamespace(observation_noise=observation_noise)
    fake_enn_pkg.EpistemicNearestNeighbors = object
    fake_enn_pkg.enn_fit = lambda *a, **k: object()
    sys.modules["third_party"] = fake_tp
    sys.modules["third_party.enn"] = fake_enn_pkg
    sys.modules["third_party.enn.enn"] = fake_enn_subpkg
    sys.modules["third_party.enn.enn.enn_params"] = fake_enn_params
    from experiments.enn import compare_to_gp as cgp
    from figures.mtv import fig_util

    class _PosteriorDist:
        def log_prob(self, y):
            return torch.tensor(float(y.shape[0]), dtype=torch.float64)

    class _Posterior:
        distribution = _PosteriorDist()

    class _Model:
        def posterior(self, x):
            return _Posterior()

    gp_ll = cgp.compute_gp_ll(_Model(), np.zeros((3, 2)), np.zeros(3))
    assert gp_ll == 1.0

    fake_fit = types.ModuleType("third_party.enn.enn.enn_fit")
    fake_fit._compute_single_loglik = lambda y, mu, se: 6.0
    sys.modules["third_party.enn.enn.enn_fit"] = fake_fit

    class _EnnModel:
        def posterior(self, test_x, params, flags):
            _ = (params, flags)
            return SimpleNamespace(mu=np.zeros((len(test_x), 1)), se=np.ones((len(test_x), 1)))

    enn_ll = cgp.compute_enn_ll(_EnnModel(), object(), np.zeros((3, 2)), np.zeros(3))
    assert enn_ll == 2.0
    assert np.isfinite(cgp.compute_mean_ll(np.array([0.0, 1.0])))
    monkeypatch.setattr(cgp, "tqdm", lambda x, desc=None: x)
    monkeypatch.setattr(cgp, "_run_dim_rep", lambda *args, **kwargs: cgp._LLResult(gp_ll=1.0, enn_ll=2.0, mean_ll=3.0))
    df = cgp.sweep_dim_ll_gp_vs_enn("sphere", 0.1, [2, 3], 0, 2, 4, 2, 3)
    assert set(df["num_dim"].tolist()) == {2, 3}

    monkeypatch.setattr(fig_util, "get_env_conf", lambda *a, **k: "env_conf")
    monkeypatch.setattr(fig_util, "default_policy", lambda env_conf: "policy")
    ep = fig_util.expository_problem()
    assert ep.opt_name == "mtv"
    assert isinstance(fig_util.show(torch.tensor([1.0, 2.0])), str)
    mesh = fig_util.mk_mesh(n=4)
    fig_util.dump_mesh(str(tmp_path), "mesh.txt", mesh.x_1, mesh.x_2, np.zeros_like(mesh.x_1))

    class _Env:
        def step(self, x):
            return None, float(np.sum(x)), False, False

    class _EnvConf:
        def make(self):
            return _Env()

    class _Post:
        def __init__(self, n):
            self.mean = torch.zeros((n, 1))
            self.variance = torch.ones((n, 1))
            self._n = n

        def sample(self, size):
            return torch.zeros(size + torch.Size([self._n]))

    class _GP:
        def posterior(self, xs):
            return _Post(len(xs))

    fig_util.mean_func_contours(str(tmp_path), _EnvConf())
    fig_util.mean_gp_contours(str(tmp_path), _GP())
    fig_util.var_contours(str(tmp_path), _GP())
    fig_util.pmax_contours(str(tmp_path), _GP())


def test_kiss_cov_fig_pstar_scale_and_turbo_best_datum(monkeypatch, tmp_path):
    from figures.pts import fig_pstar_scale as fps
    from optimizer.turbo_enn_designer import TurboENNDesigner

    monkeypatch.setattr(fps, "_num_dims", [2])
    d_args = fps.dist_pstar_scales_all_funcs("mtv", 2)
    assert d_args

    class _DM:
        def __init__(self, app_name, fn_name, job_fn):
            _ = (app_name, fn_name, job_fn)

        def __call__(self, all_args):
            assert all_args

    monkeypatch.setattr(fps, "DistModal", _DM)
    monkeypatch.setattr(fps, "dist_pstar_scales_all_funcs", lambda designer, num_dim: [{"x": 1}])
    fps.distribute("mtv", "jobs.txt", dry_run=False)

    monkeypatch.setattr(fps, "collect", lambda job_fn, cb: cb(("designer", 2, "f:sphere-2d", [("x", 1)])))
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
