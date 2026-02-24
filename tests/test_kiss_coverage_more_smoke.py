from __future__ import annotations

import json
import os
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


def test_kiss_cov_metric_and_multi_turbo_units():
    from types import SimpleNamespace

    from optimizer.multi_turbo_enn_allocation import allocated_proposal_plan
    from optimizer.multi_turbo_enn_scoring import score_multi_candidates
    from optimizer.multi_turbo_enn_state import load_multi_state
    from optimizer.multi_turbo_enn_utils import call_multi_designer
    from optimizer.trust_region_config import MetricShapedTRConfig

    cfg_metric = MetricShapedTRConfig(geometry="enn_metric_shaped", metric_sampler="full")
    assert cfg_metric.length_init > 0.0
    assert cfg_metric.length_min > 0.0
    assert cfg_metric.length_max > cfg_metric.length_min

    tr_metric = cfg_metric.build(num_dim=3, rng=np.random.default_rng(10))
    tr_metric.observe_local_geometry(
        delta_x=np.eye(3, dtype=float),
        weights=np.ones((3,), dtype=float),
    )
    tr_metric.restart()

    cfg_grad = MetricShapedTRConfig(geometry="enn_grad_metric_shaped")
    tr_grad = cfg_grad.build(num_dim=3, rng=np.random.default_rng(11))
    tr_grad.observe_local_geometry(
        delta_x=np.eye(3, dtype=float),
        weights=np.ones((3,), dtype=float),
        delta_y=np.array([1.0, 0.5, -0.25], dtype=float),
    )

    cfg_ell = MetricShapedTRConfig(geometry="enn_true_ellipsoid", metric_sampler="full")
    tr_ell = cfg_ell.build(num_dim=3, rng=np.random.default_rng(12))
    tr_ell.restart()
    x0 = np.array([0.2, 0.2, 0.2], dtype=float)
    x1 = np.array([0.3, 0.2, 0.2], dtype=float)
    tr_ell.observe_incumbent_transition(
        x_center=x0,
        y_value=1.0,
        predict_delta=lambda _prev, _cur: 1.0,
    )
    tr_ell.observe_incumbent_transition(
        x_center=x1,
        y_value=2.0,
        predict_delta=lambda _prev, _cur: 1.0,
    )

    plan = allocated_proposal_plan(
        num_arms=4,
        num_regions=2,
        pool_multiplier=2,
        allocated_num_arms=None,
        proposal_per_region=None,
    )
    assert plan.allocated_num_arms == 4
    assert plan.proposal_per_region > 0
    assert len(plan.per_region) == 2

    class _Child:
        def predict_mu_sigma(self, x):
            n = x.shape[0]
            return np.zeros((n,), dtype=float), np.ones((n,), dtype=float) * 0.1

        def best_datum(self):
            return None

    x_all = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    scores = score_multi_candidates(
        x_all,
        [0, 0],
        child_designers=[_Child()],
        region_data_lens=[1],
        region_rngs=[np.random.default_rng(13)],
        acq_type="ucb",
        rng=np.random.default_rng(14),
    )
    assert scores.shape == (2,)

    class _CallPolicy:
        def __init__(self, x):
            self._x = np.asarray(x, dtype=float)

        def get_params(self):
            return self._x

    class _CallChild:
        def __call__(self, _data, n):
            return [_CallPolicy([0.1, 0.2]) for _ in range(int(n))]

    call_state = SimpleNamespace(
        region_data=[[]],
        shared_prefix_len=0,
        region_assignments=[],
        last_region_indices=None,
        num_told_global=0,
        allocated_num_arms=None,
        proposal_per_region=None,
    )
    call_designer = SimpleNamespace(
        _tr_type="morbo",
        _arm_mode="split",
        _num_regions=1,
        _pool_multiplier=2,
        _designers=[_CallChild()],
        _state=call_state,
        _acq_type="ucb",
        _rng=np.random.default_rng(15),
        _region_rngs=[np.random.default_rng(16)],
        _init_regions=lambda _data, _num_arms: None,
        _set_telemetry=lambda _telemetry: None,
    )
    policies = call_multi_designer(call_designer, [], num_arms=1)
    assert len(policies) == 1

    load_state = SimpleNamespace(
        region_data=[],
        shared_prefix_len=0,
        region_assignments=[],
        last_region_indices=None,
        num_told_global=0,
        allocated_num_arms=None,
        proposal_per_region=None,
    )

    def _init_regions_for_load(_data, _num_arms):
        load_designer._region_rngs = [np.random.default_rng(17)]
        load_state.region_data = [[]]

    load_designer = SimpleNamespace(
        _rng=np.random.default_rng(18),
        _region_rngs=[],
        _num_regions=1,
        _strategy="shared_data",
        _designers=[],
        _state=load_state,
        _init_regions=_init_regions_for_load,
    )
    load_multi_state(
        load_designer,
        {
            "shared_prefix_len": 1,
            "num_told_global": 2,
            "region_assignments": [],
            "last_region_indices": [0],
            "allocated_num_arms": 2,
            "proposal_per_region": 4,
            "region_states": [],
        },
        data=["a", "b"],
    )
    assert load_designer._state.num_told_global == 2


def test_kiss_cov_trust_region_internal_units():
    from enn.turbo.config.candidate_rv import CandidateRV

    from optimizer.box_trust_region import FixedLengthTurboTrustRegion
    from optimizer.metric_trust_region import ENNMetricShapedTrustRegion, MetricShapedTrustRegion
    from optimizer.pc_rotation import PCRotationResult
    from optimizer.trust_region_config import MetricShapedTRConfig
    from optimizer.trust_region_utils import (
        _AxisAlignedStepSampler,
        _LengthPolicy,
        _MetricGeometryModel,
        _OptionCLengthPolicy,
        _TrueEllipsoidStepSampler,
    )

    base_policy = _LengthPolicy()
    assert base_policy.pending_rho is None
    assert (
        base_policy.apply_after_super_update(
            current_length=1.23,
            base_length=1.0,
            fixed_length=None,
            length_max=2.0,
        )
        == 1.23
    )

    option_c = _OptionCLengthPolicy(rho_bad=0.25, rho_good=0.75, gamma_down=0.5, gamma_up=2.0)
    option_c.set_acceptance_ratio(pred=1.0, act=-1.0, boundary_hit=False)
    assert option_c.pending_rho is not None
    shrunk = option_c.apply_after_super_update(
        current_length=1.0,
        base_length=1.0,
        fixed_length=None,
        length_max=4.0,
    )
    assert shrunk < 1.0
    assert option_c.pending_rho is None

    cfg_box = MetricShapedTRConfig(geometry="box", fixed_length=1.6)
    tr_box = cfg_box.build(num_dim=3, rng=np.random.default_rng(100))
    assert isinstance(tr_box, FixedLengthTurboTrustRegion)
    tr_box.update(np.array([0.0]), np.array([0.0]))
    assert np.isclose(float(tr_box.length), 1.6)
    tr_box.restart(rng=np.random.default_rng(101))
    assert np.isclose(float(tr_box.length), 1.6)

    cfg_metric = MetricShapedTRConfig(geometry="enn_metric_shaped", metric_sampler="full")
    tr_metric = cfg_metric.build(num_dim=3, rng=np.random.default_rng(102))
    assert isinstance(tr_metric, MetricShapedTrustRegion)
    assert isinstance(tr_metric, ENNMetricShapedTrustRegion)
    tr_metric.set_gradient_geometry(
        delta_x=np.eye(3, dtype=float),
        delta_y=np.array([1.0, 0.5, 0.25], dtype=float),
        weights=np.ones((3,), dtype=float),
    )

    model = _MetricGeometryModel(
        num_dim=3,
        metric_sampler="full",
        metric_rank=None,
        pc_rotation_mode="full",
        pc_rank=None,
    )
    delta_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    weights = np.array([1.0, 2.0, 1.0], dtype=float)
    model.set_gradient_geometry(
        delta_x=delta_x,
        delta_y=np.array([1.0, 0.5, 0.2], dtype=float),
        weights=weights,
    )
    centered = delta_x - np.mean(delta_x, axis=0, keepdims=True)
    model.update_from_cov(
        centered=centered,
        weights=np.array([0.25, 0.5, 0.25], dtype=float),
        cov=np.eye(3, dtype=float),
    )
    model.update_from_pc_rotation(
        PCRotationResult(
            center=np.zeros((3,), dtype=float),
            basis=np.eye(3, dtype=float),
            singular_values=np.array([2.0, 1.0, 0.5], dtype=float),
            has_rotation=True,
        )
    )
    cov = model.covariance_matrix(jitter=1e-6)
    assert cov.shape == (3, 3)
    step = model.build_step(np.zeros((4, 3), dtype=float), np.random.default_rng(103))
    assert step.shape == (4, 3)

    axis_sampler = _AxisAlignedStepSampler(default_candidate_rv=CandidateRV.UNIFORM)
    axis_candidates = axis_sampler.generate(
        x_center=np.full((3,), 0.5, dtype=float),
        length=0.5,
        num_dim=3,
        num_candidates=8,
        rng=np.random.default_rng(104),
        candidate_rv=CandidateRV.UNIFORM,
        sobol_engine=None,
        num_pert=2,
        build_step=lambda z, _rng: z,
    )
    assert axis_candidates.shape == (8, 3)

    ellipsoid_sampler = _TrueEllipsoidStepSampler(
        default_candidate_rv=CandidateRV.SOBOL,
        p_raasp=0.4,
        radial_mode="boundary",
    )
    ellipsoid_candidates = ellipsoid_sampler.generate(
        x_center=np.full((3,), 0.5, dtype=float),
        num_dim=3,
        num_candidates=8,
        length=0.25,
        rng=np.random.default_rng(105),
        candidate_rv=CandidateRV.UNIFORM,
        covariance_matrix=np.eye(3, dtype=float),
    )
    assert ellipsoid_candidates.shape == (8, 3)
    assert np.all(ellipsoid_candidates >= -1e-10)
    assert np.all(ellipsoid_candidates <= 1.0 + 1e-10)


def test_kiss_cov_env_conf_atari_dm():
    """Cover env_conf_atari_dm lazy-load helpers."""
    from problems.env_conf_atari_dm import (
        get_atari_make,
        get_atari_parsers_and_factories,
        get_cnn_mlp_policy_factory,
        get_dm_control_make,
    )

    cnn_factory = get_cnn_mlp_policy_factory()
    assert cnn_factory is not None

    parse_fn, agent57_f, cnn_f, gauss_f = get_atari_parsers_and_factories()
    assert parse_fn is not None
    assert agent57_f is not None
    assert cnn_f is not None
    assert gauss_f is not None

    make_dm = get_dm_control_make()
    assert callable(make_dm)

    make_atari = get_atari_make()
    assert callable(make_atari)
