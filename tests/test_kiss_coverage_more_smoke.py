from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import torch
from kiss_more_smoke_acq_gp import _AcqBTGP, _AcqDPPModel
from kiss_more_smoke_exp_dist import _DistFactory
from kiss_more_smoke_fig_env import _FigEnvConf
from kiss_more_smoke_fig_gp import _FigGP
from kiss_more_smoke_modal_func import ModalBatchesSpawnFunc
from kiss_more_smoke_noise_policy import _DummyEnv, _NoisePolicy
from kiss_more_smoke_pstar_dm import _PstarDistModal
from torch import nn

from acq.acq_bt import AcqBT
from acq.acq_dpp import AcqDPP
from acq.fit_gp import _EmptyTransform
from optimizer.bt_designer import BTDesigner
from optimizer.designer_registry_context import _SimpleContext
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
from policies.policy_mixin import PolicyParamsMixin
from problems.noise_maker import NoiseMaker


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
    monkeypatch.setattr("acq.acq_bt.fit_gp.fit_gp_XY", lambda X, Y, model_spec: _AcqBTGP())
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
    acq = AcqDPP(_AcqDPPModel(), num_X_samples=8, num_runs=1)
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
    from tests.kiss_ops_catalog_data_shared import run_kiss_data_locator_optimizers

    run_kiss_data_locator_optimizers(tmp_path)


def test_kiss_cov_ops_catalog_and_data_cli(tmp_path):
    from tests.kiss_ops_catalog_data_shared import run_kiss_ops_catalog_and_data_cli

    run_kiss_ops_catalog_and_data_cli(tmp_path)


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
    noise = NoiseMaker(_DummyEnv(), normalized_noise_level=0.0, num_measurements=2)
    assert noise.observation_space is not None
    assert noise.action_space is not None
    assert NoiseMaker.observation_space.fget(noise) is not None
    assert NoiseMaker.action_space.fget(noise) is not None

    p = _NoisePolicy()
    assert p.num_params() == 2
    assert PolicyParamsMixin.num_params(p) == 2


def test_kiss_cov_exp_uhd_cli_and_local(monkeypatch, tmp_path):
    import ops.exp_uhd as exp_uhd
    import ops.uhd_setup_make_loop as uhd_setup

    called = {"run": 0}

    monkeypatch.setattr(
        uhd_setup,
        "make_loop",
        lambda *a, **k: SimpleNamespace(run=lambda: called.__setitem__("run", called["run"] + 1)),
    )
    exp_uhd.cli.callback()
    toml_file = tmp_path / "test.toml"
    toml_file.write_text('[uhd]\nenv_tag = "f:sphere-2d"\npolicy_tag = "pure-function"\nnum_rounds = 1\n')
    exp_uhd.local.callback(str(toml_file))
    assert called["run"] == 1


def test_kiss_cov_dist_modal_collect(monkeypatch, tmp_path):
    import experiments.dist_modal as dist_modal

    monkeypatch.setattr(dist_modal.modal.functions, "FunctionCall", _DistFactory)
    fn = tmp_path / "jobs.txt"
    fn.write_text("abc\n")
    got = []
    dist_modal.collect(str(fn), lambda x: got.append(x))
    assert got == [{"ok": True}]


def test_kiss_cov_modal_batches_functions(monkeypatch, tmp_path):
    from modal_timing_sweep_test_support import FakeResultsDict

    import experiments.modal_batches_impl as mb

    tag = "test"

    res_dict = FakeResultsDict()
    submitted = FakeResultsDict()
    monkeypatch.setattr(mb, "_results_dict", lambda _tag: res_dict)
    monkeypatch.setattr(mb, "_submitted_dict", lambda _tag: submitted)
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

    spawned = {"map": [], "spawn": []}

    monkeypatch.setattr(
        mb.modal.Function,
        "from_name",
        lambda app_name, name: ModalBatchesSpawnFunc(spawned),
    )
    monkeypatch.setattr(mb, "_gen_jobs", lambda _batch_tag: [("k1", SimpleNamespace(trace_fn="t1"))])
    monkeypatch.setattr(mb, "data_is_done", lambda trace_fn: False)
    monkeypatch.setattr(mb, "post_process", lambda *args, **kwargs: None)

    mb.modal_batches_worker.get_raw_f()((tag, "k0", SimpleNamespace(trace_fn="trace0")))
    assert "k0" in res_dict

    mb.modal_batches_resubmitter.get_raw_f()([("k1", SimpleNamespace(trace_fn="t1"), False)], tag)
    assert submitted["k1"] is True

    mb.batches_submitter(tag, "batch_tag")
    assert spawned["spawn"]

    res_dict["k2"] = ("trace_fn", "log", "trace", None)
    mb._collect(tag)
    mb.status(tag)

    mb.modal_batch_deleter.get_raw_f()(["k2"], tag)
    assert "k2" not in res_dict

    deleted = []
    monkeypatch.setattr(mb.modal.Dict, "delete", lambda name: deleted.append(name))
    mb.clean_up(tag)
    assert "batches_dict_test" in deleted
    assert "submitted_dict_test" in deleted


def test_kiss_cov_modal_batches_clean_up_exception(monkeypatch, capsys):
    import experiments.modal_batches_impl as mb

    def _raise_error(name):
        raise RuntimeError(f"Failed to delete {name}")

    monkeypatch.setattr(mb.modal.Dict, "delete", _raise_error)
    mb.clean_up("test")

    captured = capsys.readouterr()
    assert "CLEANUP: dict delete failed" in captured.out
    assert "batches_dict_test" in captured.out
    assert "submitted_dict_test" in captured.out


def test_kiss_cov_fig_util_functions(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from figures.mtv import fig_util

    fake_env = SimpleNamespace()
    fake_problem = SimpleNamespace(env=fake_env, build_policy=lambda: "policy")
    monkeypatch.setattr(fig_util, "build_problem", lambda *a, **k: fake_problem)
    ep = fig_util.expository_problem()
    assert ep.opt_name == "mtv"
    assert isinstance(fig_util.show(torch.tensor([1.0, 2.0])), str)

    mesh = fig_util.mk_mesh(n=4)
    assert mesh.xs.shape[1] == 2
    fig_util.dump_mesh(str(tmp_path), "mesh.txt", mesh.x_1, mesh.x_2, np.zeros_like(mesh.x_1))
    assert (tmp_path / "mesh.txt").exists()

    fig_util.mean_func_contours(str(tmp_path), _FigEnvConf())
    fig_util.mean_gp_contours(str(tmp_path), _FigGP())
    fig_util.var_contours(str(tmp_path), _FigGP())
    fig_util.pmax_contours(str(tmp_path), _FigGP())
    assert (tmp_path / "mean_func").exists()


def test_kiss_cov_fig_pstar_scale_functions(monkeypatch, tmp_path):
    from figures.pts import fig_pstar_scale as fps

    monkeypatch.setattr(fps, "_num_dims", [2])
    d_args = fps.dist_pstar_scales_all_funcs("mtv", 2)
    assert d_args

    called = {"dist": 0, "collect": 0}

    _PstarDistModal.hook(called)
    monkeypatch.setattr(fps, "DistModal", _PstarDistModal)
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
