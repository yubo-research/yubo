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
from optimizer.gaussian_perturbator import GaussianPerturbator
from optimizer.lr_scheduler import (
    ConstantLR,
    LinearLRScheduler,
    LRScheduler,
    OneCycleLR,
)
from optimizer.uhd_bgd import UHDBGD
from optimizer.uhd_hoeffding import UHDHoeffding
from optimizer.uhd_mezo import UHDMeZO
from optimizer.uhd_simple import UHDSimple
from problems.noise_maker import NoiseMaker
from problems.policy_mixin import PolicyParamsMixin


def _make_gp():
    module = nn.Linear(3, 2, bias=True)
    dim = sum(p.numel() for p in module.parameters())
    return module, GaussianPerturbator(module), dim


def test_kiss_cov_lr_scheduler_protocol_and_lr_properties():
    c = ConstantLR(0.01)
    linear_sched = LinearLRScheduler(0.1, num_steps=10, warmup_steps=2)
    one_cycle_sched = OneCycleLR(0.1, num_steps=10)
    assert isinstance(c, LRScheduler)
    assert isinstance(linear_sched, LRScheduler)
    assert isinstance(one_cycle_sched, LRScheduler)
    assert c.lr > 0
    assert linear_sched.lr >= 0
    assert one_cycle_sched.lr > 0
    linear_sched.step()
    one_cycle_sched.step()
    assert linear_sched.lr >= 0
    assert one_cycle_sched.lr > 0
    assert isinstance(LRScheduler.lr, property)
    assert ConstantLR.lr.fget(c) > 0
    assert LinearLRScheduler.lr.fget(linear_sched) >= 0
    assert OneCycleLR.lr.fget(one_cycle_sched) > 0


def test_kiss_cov_uhd_simple_properties():
    _, gp, dim = _make_gp()
    uhd = UHDSimple(gp, sigma_0=0.1, dim=dim)
    assert uhd.eval_seed == 0
    assert uhd.y_best is None
    assert uhd.mu_avg == 0.0
    assert uhd.se_avg == 0.0
    uhd.ask()
    uhd.tell(1.5, 0.2)
    assert uhd.eval_seed == 1
    assert uhd.y_best == 1.5
    assert uhd.mu_avg == 1.5
    assert uhd.se_avg == 0.2
    assert UHDSimple.eval_seed.fget(uhd) == 1
    assert UHDSimple.y_best.fget(uhd) == 1.5
    assert UHDSimple.mu_avg.fget(uhd) == 1.5
    assert UHDSimple.se_avg.fget(uhd) == 0.2


def test_kiss_cov_uhd_mezo_properties():
    _, gp, dim = _make_gp()
    uhd = UHDMeZO(gp, dim=dim, lr_scheduler=ConstantLR(0.01), sigma=0.1)
    assert uhd.eval_seed == 0
    assert uhd.y_best is None
    assert uhd.mu_avg == 0.0
    assert uhd.se_avg == 0.0
    uhd.ask()
    uhd.tell(2.0, 0.3)
    uhd.ask()
    uhd.tell(1.0, 0.1)
    assert uhd.eval_seed == 1
    assert uhd.y_best == 2.0
    assert uhd.mu_avg == 1.0
    assert uhd.se_avg == 0.1
    assert UHDMeZO.eval_seed.fget(uhd) == 1
    assert UHDMeZO.y_best.fget(uhd) == 2.0
    assert UHDMeZO.mu_avg.fget(uhd) == 1.0
    assert UHDMeZO.se_avg.fget(uhd) == 0.1


def test_kiss_cov_uhd_bgd_properties():
    _, gp, dim = _make_gp()
    uhd = UHDBGD(gp, sigma_0=0.1, dim=dim)
    assert uhd.y_best is None
    assert uhd.mu_avg == 0.0
    assert uhd.se_avg == 0.0
    uhd.ask()
    uhd.tell(1.0, 0.05)
    assert uhd.y_best == 1.0
    assert uhd.mu_avg == 1.0
    assert uhd.se_avg == 0.05
    assert UHDBGD.y_best.fget(uhd) == 1.0
    assert UHDBGD.mu_avg.fget(uhd) == 1.0
    assert UHDBGD.se_avg.fget(uhd) == 0.05


def test_kiss_cov_uhd_hoeffding_properties():
    _, gp, dim = _make_gp()
    uhd = UHDHoeffding(gp, sigma_0=0.1, dim=dim)
    assert uhd.eval_seed == 0
    assert uhd.y_best is None
    uhd.ask()
    uhd.tell(1.0, 0.05)
    assert uhd.y_best is not None
    assert uhd.mu_avg > 0.0
    assert uhd.se_avg >= 0.0
    assert UHDHoeffding.eval_seed.fget(uhd) >= 0
    assert UHDHoeffding.y_best.fget(uhd) is not None
    assert UHDHoeffding.mu_avg.fget(uhd) > 0.0
    assert UHDHoeffding.se_avg.fget(uhd) >= 0.0


def test_kiss_cov_acqbt_x_max(monkeypatch):
    class _GP:
        def __call__(self, _x):
            return SimpleNamespace(mean=torch.tensor(0.0))

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


def test_kiss_cov_acq_dpp_init():
    class _Model:
        def __init__(self):
            self.train_inputs = (torch.zeros(2, 3, dtype=torch.double),)
            self.likelihood = SimpleNamespace(noise=torch.tensor(1.0, dtype=torch.double))

        def eval(self):
            return None

    acq = AcqDPP(_Model(), num_X_samples=8, num_runs=1)
    assert acq._num_dim == 3


def test_kiss_cov_empty_transform_init():
    t = _EmptyTransform()
    y = torch.tensor([[1.0]])
    y2, yvar2 = t.forward(y)
    assert torch.equal(y2, y)
    assert yvar2 is None
    y3, yvar3 = t.untransform(y)
    assert torch.equal(y3, y)
    assert yvar3 is None
    assert t.untransform_posterior("p") == "p"


def test_kiss_cov_data_locator_optimizers(tmp_path):
    results_dir = tmp_path / "results"
    exp_dir = results_dir / "exp_a"
    exp_dir.mkdir(parents=True)
    (exp_dir / "config.json").write_text(json.dumps({"opt_name": "random", "env_tag": "f:ackley-2d"}))
    dl = DataLocator(
        results_path=str(results_dir),
        exp_dir="",
        opt_names=["random", "sobol"],
    )
    assert dl.optimizers() == ["random"]


def test_kiss_cov_ops_catalog_and_data_cli(tmp_path):
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
    (exp_dir / "config.json").write_text(
        json.dumps(
            {
                "opt_name": "random",
                "env_tag": "f:ackley-2d",
                "num_arms": 1,
                "num_rounds": 1,
            }
        )
    )
    (traces / "00000.jsonl").write_text("{}\n")

    res = runner.invoke(data_cli.cli, ["ls", str(results_dir)])
    assert res.exit_code == 0
    assert "abc123" in res.output
    data_cli.cli.callback()
    data_cli.ls.callback(results_dir, False)

    res = runner.invoke(data_cli.cli, ["rm", str(results_dir), "abc123", "-f"])
    assert res.exit_code == 0
    assert not exp_dir.exists()

    exp_dir.mkdir(parents=True)
    (exp_dir / "config.json").write_text(json.dumps({"opt_name": "random", "env_tag": "f:ackley-2d"}))
    data_cli.rm.callback(results_dir, ("abc123",), True)
    assert not exp_dir.exists()


def test_kiss_cov_context_and_best_datum():
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
    assert NoiseMaker.observation_space.fget(noise) is not None
    assert NoiseMaker.action_space.fget(noise) is not None

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
    assert PolicyParamsMixin.num_params(p) == 2


def test_kiss_cov_exp_uhd_cli_and_local(monkeypatch, tmp_path):
    import ops.exp_uhd as exp_uhd
    import ops.uhd_setup as uhd_setup

    called = {"run": 0}

    class _Loop:
        def run(self):
            called["run"] += 1

    monkeypatch.setattr(uhd_setup, "make_loop", lambda *a, **k: _Loop())
    exp_uhd.cli.callback()
    toml_file = tmp_path / "test.toml"
    toml_file.write_text('[uhd]\nenv_tag = "f:sphere-2d"\nnum_rounds = 1\n')
    exp_uhd.local.callback(str(toml_file))
    assert called["run"] == 1


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


def test_kiss_cov_modal_batches_functions(monkeypatch, tmp_path):
    import experiments.modal_batches as mb

    class _FakeDict(dict):
        def len(self):
            return len(self)

    res_dict = _FakeDict()
    submitted = _FakeDict()
    monkeypatch.setattr(mb, "_results_dict", lambda: res_dict)
    monkeypatch.setattr(mb, "_submitted_dict", lambda: submitted)
    monkeypatch.setattr(mb, "sample_1", lambda run_cfg: ("log", "trace", [{"x": 1}]))

    spawned = {"map": [], "spawn": []}

    class _Func:
        def spawn_map(self, todo):
            spawned["map"].append(list(todo))

        def spawn(self, payload):
            spawned["spawn"].append(payload)

    monkeypatch.setattr(mb.modal.Function, "from_name", lambda app_name, name: _Func())
    monkeypatch.setattr(mb, "_gen_jobs", lambda tag: [("k1", SimpleNamespace(trace_fn="t1"))])
    monkeypatch.setattr(mb, "data_is_done", lambda trace_fn: False)
    monkeypatch.setattr(mb, "post_process", lambda *args, **kwargs: None)

    mb.modal_batches_worker.get_raw_f()(("k0", SimpleNamespace(trace_fn="trace0")))
    assert "k0" in res_dict

    mb.modal_batches_resubmitter.get_raw_f()([("k1", SimpleNamespace(trace_fn="t1"), False)])
    assert submitted["k1"] is True

    mb.batches_submitter("tag")
    assert spawned["spawn"]

    res_dict["k2"] = ("trace_fn", "log", "trace", None)
    mb.collect()
    mb.status()

    mb.modal_batch_deleter.get_raw_f()(["k2"])
    assert "k2" not in res_dict

    deleted = []
    monkeypatch.setattr(mb.modal.Dict, "delete", lambda name: deleted.append(name))
    mb.clean_up()
    assert "batches_dict" in deleted


def test_kiss_cov_compare_to_gp_functions(monkeypatch):
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

    class _PosteriorDist:
        def log_prob(self, y):
            return torch.tensor(float(y.shape[0]), dtype=torch.float64)

    class _Posterior:
        distribution = _PosteriorDist()

    class _Model:
        def posterior(self, x):
            assert x.ndim == 2
            return _Posterior()

    gp_ll = cgp.compute_gp_ll(_Model(), np.zeros((3, 2)), np.zeros(3))
    assert gp_ll == 1.0

    fake_mod = types.ModuleType("third_party.enn.enn.enn_fit")
    fake_mod._compute_single_loglik = lambda y, mu, se: 6.0
    sys.modules["third_party.enn.enn.enn_fit"] = fake_mod

    class _EnnModel:
        def posterior(self, test_x, params, flags):
            _ = (params, flags)
            return SimpleNamespace(mu=np.zeros((len(test_x), 1)), se=np.ones((len(test_x), 1)))

    enn_ll = cgp.compute_enn_ll(_EnnModel(), object(), np.zeros((3, 2)), np.zeros(3))
    assert enn_ll == 2.0
    assert np.isfinite(cgp.compute_mean_ll(np.array([0.0, 1.0])))

    monkeypatch.setattr(cgp, "tqdm", lambda x, desc=None: x)
    monkeypatch.setattr(
        cgp,
        "_run_dim_rep",
        lambda *args, **kwargs: cgp._LLResult(gp_ll=1.0, enn_ll=2.0, mean_ll=3.0),
    )
    df = cgp.sweep_dim_ll_gp_vs_enn("sphere", 0.1, [2, 3], 0, 2, 4, 2, 3)
    assert set(df["num_dim"].tolist()) == {2, 3}


def test_kiss_cov_fig_util_functions(monkeypatch, tmp_path):
    from figures.mtv import fig_util

    monkeypatch.setattr(fig_util, "get_env_conf", lambda *a, **k: "env_conf")
    monkeypatch.setattr(fig_util, "default_policy", lambda env_conf: "policy")
    ep = fig_util.expository_problem()
    assert ep.opt_name == "mtv"
    assert isinstance(fig_util.show(torch.tensor([1.0, 2.0])), str)

    mesh = fig_util.mk_mesh(n=4)
    assert mesh.xs.shape[1] == 2
    fig_util.dump_mesh(str(tmp_path), "mesh.txt", mesh.x_1, mesh.x_2, np.zeros_like(mesh.x_1))
    assert (tmp_path / "mesh.txt").exists()

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
    assert (tmp_path / "mean_func").exists()


def test_kiss_cov_fig_pstar_scale_functions(monkeypatch, tmp_path):
    from figures.pts import fig_pstar_scale as fps

    monkeypatch.setattr(fps, "_num_dims", [2])
    d_args = fps.dist_pstar_scales_all_funcs("mtv", 2)
    assert d_args

    called = {"dist": 0, "collect": 0}

    class _DM:
        def __init__(self, app_name, fn_name, job_fn):
            _ = (app_name, fn_name, job_fn)

        def __call__(self, all_args):
            called["dist"] += len(all_args)

    monkeypatch.setattr(fps, "DistModal", _DM)
    monkeypatch.setattr(fps, "dist_pstar_scales_all_funcs", lambda designer, num_dim: [{"x": 1}])
    fps.distribute("mtv", "jobs.txt", dry_run=False)
    assert called["dist"] == 1

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
    assert (tmp_path / "fig_data" / "sts").exists()

    monkeypatch.setattr(
        fps,
        "distribute",
        lambda *a, **k: called.__setitem__("collect", called["collect"] + 1),
    )
    monkeypatch.setattr(
        fps,
        "collect_all",
        lambda *a, **k: called.__setitem__("collect", called["collect"] + 1),
    )
    fps.spawn_all("dist", "jobs.txt", False, "mtv")
    fps.spawn_all("collect", "jobs.txt", False, "mtv")
    assert called["collect"] >= 2
