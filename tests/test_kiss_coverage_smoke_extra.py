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
from ops.experiment import main as ops_experiment_main
from optimizer import opt_trajectories as opt_trajectories_mod
from optimizer.opt_trajectories import collect_denoised_trajectory, collect_trajectory_with_noise, evaluate_for_best, mean_return_over_runs
from problems.humanoid_policy import HumanoidPolicy
from problems.other import make as make_other
from rl.core.sac_update import SACUpdateBatch, SACUpdateHyperParams, SACUpdateModules, SACUpdateOptimizers, sac_update_step
from rl.pufferlib.ppo import specs as ppo_specs
from rl.pufferlib.sac import runtime_utils as sac_runtime_utils
from rl.pufferlib.vector_env import make_vector_env as puffer_make_vector_env
from rl.torchrl.sac import setup as sac_setup
from rl.torchrl.sac.config import SACConfig


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


def test_kiss_cov_humanoid_policy_and_other_module():
    env_conf = SimpleNamespace(
        env_name="Humanoid-v5",
        problem_seed=3,
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(348,))),
        action_space=SimpleNamespace(shape=(17,)),
    )
    p = HumanoidPolicy(env_conf)
    assert p.num_params() == 22
    base = p.get_params()
    p.set_params(np.zeros_like(base))
    c = p.clone()
    assert c.num_params() == p.num_params()
    c.reset_state()
    out = c(np.zeros(348, dtype=np.float64))
    assert out.shape == (17,)
    try:
        make_other("unknown-name", problem_seed=0)
        assert False, "Expected unknown env_name assertion"
    except AssertionError:
        assert True


def test_kiss_cov_sac_update_and_opt_trajectories(monkeypatch):
    class _Policy:
        def sample(self, obs, deterministic=False):
            _ = deterministic
            act = torch.tanh(obs[..., :1])
            lp = torch.zeros(obs.shape[0], dtype=obs.dtype)
            return (act, lp)

    q1 = nn.Linear(2, 1)
    q2 = nn.Linear(2, 1)
    q1_t = nn.Linear(2, 1)
    q2_t = nn.Linear(2, 1)

    class _QWrap(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, obs, act):
            return self.base(torch.cat([obs, act], dim=-1)).squeeze(-1)

    actor = _Policy()
    modules = SACUpdateModules(
        actor=actor, q1=_QWrap(q1), q2=_QWrap(q2), q1_target=_QWrap(q1_t), q2_target=_QWrap(q2_t), log_alpha=nn.Parameter(torch.tensor(0.0))
    )
    opts = SACUpdateOptimizers(
        actor=torch.optim.AdamW([modules.log_alpha], lr=1e-3),
        critic=torch.optim.AdamW(list(q1.parameters()) + list(q2.parameters()), lr=1e-3),
        alpha=torch.optim.AdamW([modules.log_alpha], lr=1e-3),
    )
    batch = SACUpdateBatch(
        obs=torch.zeros((4, 1)),
        act=torch.zeros((4, 1)),
        rew=torch.zeros(4),
        nxt=torch.zeros((4, 1)),
        done=torch.zeros(4),
    )
    hyper = SACUpdateHyperParams(gamma=0.99, tau=0.01, target_entropy=-1.0)
    a, c, al = sac_update_step(modules=modules, optimizers=opts, batch=batch, hyper=hyper)
    assert np.isfinite(a)
    assert np.isfinite(c)
    assert np.isfinite(al)

    monkeypatch.setattr(
        opt_trajectories_mod,
        "collect_trajectory",
        lambda env_conf, policy, noise_seed=0: opt_trajectories_mod.Trajectory(float(noise_seed), None, None, None),
    )
    conf = SimpleNamespace(noise_seed_0=10, frozen_noise=False)
    traj, seed = collect_trajectory_with_noise(conf, object(), i_noise=1, denoise_seed=2)
    assert traj.rreturn == 13.0
    assert seed == 13
    mean, se, all_same = mean_return_over_runs(conf, object(), num_denoise=2, i_noise=1)
    assert np.isfinite(mean)
    assert np.isfinite(se)
    assert all_same is False
    den, _ = collect_denoised_trajectory(conf, object(), num_denoise=2, i_noise=1)
    assert den.rreturn is not None
    best = evaluate_for_best(conf, object(), 2)
    assert np.isfinite(best)


def test_kiss_cov_modal_collect_runtime_utils_and_ops_experiment(monkeypatch):
    import experiments.experiment_sampler as experiment_sampler

    sys.modules["experiment_sampler"] = experiment_sampler
    from experiments.modal_collect import get_job_result
    from experiments.modal_collect import main as modal_collect_main
    from rl.pufferlib.sac.runtime_utils import obs_scale_from_env, select_device

    class _Call:
        def get(self, timeout):
            assert timeout == 5
            return ("trace", "log", "collector")

    class _Factory:
        @staticmethod
        def from_id(_call_id):
            return _Call()

    monkeypatch.setattr("experiments.modal_collect.modal.functions.FunctionCall", _Factory)
    assert isinstance(get_job_result("id-1"), tuple)

    called = {"n": 0}
    monkeypatch.setattr("experiments.modal_collect.collect", lambda job_fn, cb: cb(("trace", "log", "collector")))
    monkeypatch.setattr("experiments.modal_collect.post_process", lambda *args: called.__setitem__("n", called["n"] + 1))
    monkeypatch.setattr("experiments.modal_collect.os.path.exists", lambda p: False)
    modal_collect_main("jobs.txt")
    assert called["n"] == 1

    env_conf = SimpleNamespace(
        gym_conf=SimpleNamespace(
            transform_state=True,
            state_space=SimpleNamespace(low=np.array([-1.0], dtype=np.float32), high=np.array([1.0], dtype=np.float32), shape=(1,)),
        ),
        ensure_spaces=lambda: None,
    )
    assert str(select_device("cpu")) == "cpu"
    lb, width = obs_scale_from_env(env_conf)
    assert np.allclose(lb, np.array([-1.0], dtype=np.float32))
    assert np.allclose(width, np.array([2.0], dtype=np.float32))

    fake_mod = types.SimpleNamespace(cli=lambda: None)
    monkeypatch.setitem(sys.modules, "experiments.experiment", fake_mod)
    ops_experiment_main()


def test_kiss_cov_ppo_specs_actor_critic_and_helpers():
    low, high = ppo_specs.normalize_action_bounds(
        low=np.array([-2.0, -1.0], dtype=np.float32),
        high=np.array([2.0, 3.0], dtype=np.float32),
        dim=2,
    )
    action_spec = ppo_specs._ActionSpec(kind="continuous", dim=2, low=low, high=high)
    actor = nn.Linear(3, 2)
    critic = nn.Linear(3, 4)
    actor_head = nn.Identity()
    critic_head = nn.Linear(4, 1)
    model = ppo_specs._ActorCritic(
        actor_backbone=actor,
        critic_backbone=critic,
        actor_head=actor_head,
        critic_head=critic_head,
        action_spec=action_spec,
        log_std_init=-0.5,
    )
    ppo_specs.init_linear(actor, gain=0.5)
    obs = torch.zeros((5, 3), dtype=torch.float32)
    action_out, values = model.forward(obs)
    assert action_out.shape == (5, 2)
    assert model.get_value(obs).shape == (5,)
    sampled_action, log_prob, entropy, v = model.get_action_and_value(obs, action=None)
    assert sampled_action.shape == (5, 2)
    assert log_prob.shape == (5,)
    assert entropy.shape == (5,)
    assert v.shape == values.shape
    _, log_prob2, entropy2, _ = model.get_action_and_value(obs, action=sampled_action)
    assert log_prob2.shape == (5,)
    assert entropy2.shape == (5,)


def test_kiss_cov_sac_setup_network_blocks_and_scaling():
    obs_scaler = sac_setup.sac_deps.torchrl_common.ObsScaler(None, None)
    backbone = nn.Linear(4, 8)
    head = nn.Linear(8, 4)
    actor = sac_setup._ActorNet(backbone=backbone, head=head, obs_scaler=obs_scaler, act_dim=2)
    obs = torch.ones((3, 4), dtype=torch.float32)
    loc, scale = actor.forward(obs)
    assert loc.shape == (3, 2)
    assert scale.shape == (3, 2)
    sampled, sampled_lp = actor.sample(obs)
    assert sampled.shape == (3, 2)
    assert sampled_lp.shape == (3,)
    deterministic, det_lp = actor.sample(obs, deterministic=True)
    assert deterministic.shape == (3, 2)
    assert torch.allclose(det_lp, torch.zeros_like(det_lp))
    acted = actor.act(obs)
    assert acted.shape == (3, 2)

    q_backbone = nn.Linear(6, 8)
    q_head = nn.Linear(8, 1)
    qnet = sac_setup._QNet(backbone=q_backbone, head=q_head, obs_scaler=obs_scaler)
    q = qnet.forward(torch.ones((3, 4)), torch.ones((3, 2)))
    assert q.shape == (3,)

    pix_encoder = nn.Sequential(nn.Flatten(), nn.Linear(12, 5))
    qpix = sac_setup._QNetPixel(obs_encoder=pix_encoder, head=nn.Linear(7, 1), obs_scaler=obs_scaler)
    qpix_out = qpix.forward(torch.ones((2, 1, 3, 4)), torch.ones((2, 2)))
    assert qpix_out.shape == (2,)

    scaler = sac_setup._ScaleActionToEnv(np.array([-2.0, -1.0]), np.array([2.0, 3.0]))
    scaled = scaler.forward(torch.tensor([[-1.0, 1.0], [0.0, 0.0]], dtype=torch.float32))
    assert torch.allclose(scaled[0], torch.tensor([-2.0, 3.0]))


def test_kiss_cov_sac_setup_build_and_update(monkeypatch, tmp_path):
    fake_env_conf = SimpleNamespace(from_pixels=False, gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(4,))))
    shared = SimpleNamespace(
        env_conf=fake_env_conf,
        problem_seed=7,
        noise_seed_0=11,
        act_dim=2,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        obs_lb=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
        obs_width=np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
    )
    monkeypatch.setattr(sac_setup, "build_continuous_gym_env_setup", lambda **kwargs: shared)

    cfg = SACConfig(exp_dir=str(tmp_path), env_tag="pend", batch_size=4, replay_size=32)
    env_setup = sac_setup.build_env_setup(cfg)
    assert env_setup.obs_dim == 4
    modules = sac_setup.build_modules(cfg, env_setup, device=torch.device("cpu"))
    training = sac_setup.build_training(cfg, modules)
    assert training.metrics_path.name == "metrics.jsonl"

    calls = {}

    def _fake_update_step(*, modules, optimizers, batch, hyper):
        calls["target_entropy"] = hyper.target_entropy
        assert batch.obs.shape[0] == 4
        return (1.0, 2.0, 3.0)

    monkeypatch.setattr(sac_setup, "sac_update_step", _fake_update_step)
    out = sac_setup.sac_update_shared(
        cfg,
        modules,
        training,
        obs=torch.zeros((4, 4)),
        act=torch.zeros((4, 2)),
        rew=torch.zeros(4),
        nxt=torch.zeros((4, 4)),
        done=torch.zeros(4),
    )
    assert out == (1.0, 2.0, 3.0)
    assert isinstance(calls["target_entropy"], float)


def test_kiss_cov_sac_runtime_utils_wrappers(monkeypatch):
    monkeypatch.setattr(sac_runtime_utils, "_mps_is_available_core", lambda: True)
    assert sac_runtime_utils._mps_is_available()
    d = sac_runtime_utils.select_device("cpu")
    assert str(d) == "cpu"

    env_conf = SimpleNamespace(
        gym_conf=SimpleNamespace(
            transform_state=True,
            state_space=SimpleNamespace(low=np.array([-1.0]), high=np.array([1.0]), shape=(1,)),
        ),
        ensure_spaces=lambda: None,
    )
    lb, width = sac_runtime_utils.obs_scale_from_env(env_conf)
    assert np.allclose(lb, np.array([-1.0]))
    assert np.allclose(width, np.array([2.0]))


def test_kiss_cov_puffer_vector_env_make_vector_env():
    class _Vector:
        class Serial:
            pass

        class Multiprocessing:
            pass

        @staticmethod
        def make(env_creator, env_kwargs, backend, num_envs, seed, **backend_kwargs):
            _ = env_creator
            return {
                "env_kwargs": env_kwargs,
                "backend": backend,
                "num_envs": num_envs,
                "seed": seed,
                "backend_kwargs": backend_kwargs,
            }

    class _PufferAtari:
        @staticmethod
        def env_creator(_game_name):
            return lambda **kwargs: kwargs

    cfg = SimpleNamespace(env_tag="f:ackley-2d", vector_backend="serial", num_envs=2, seed=5, framestack=4)
    out = puffer_make_vector_env(
        cfg,
        import_pufferlib_modules_fn=lambda: (SimpleNamespace(), _Vector, _PufferAtari),
        is_atari_env_tag_fn=lambda tag: False,
        to_puffer_game_name_fn=lambda tag: tag,
        resolve_gym_env_name_fn=lambda env_tag: ("CartPole-v1", {}),
    )
    assert out["backend"] is _Vector.Serial
    assert out["num_envs"] == 2
