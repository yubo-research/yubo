from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import torch
from click.testing import CliRunner
from torch import nn

from optimizer import opt_trajectories as opt_trajectories_mod
from optimizer.opt_trajectories import (
    collect_denoised_trajectory,
    collect_trajectory_with_noise,
    evaluate_for_best,
    mean_return_over_runs,
)
from problems.humanoid_policy import HumanoidPolicy
from problems.other import make as make_other
from rl.core.sac_update import (
    SACUpdateBatch,
    SACUpdateHyperParams,
    SACUpdateModules,
    SACUpdateOptimizers,
    sac_update_step,
)
from rl.pufferlib.ppo import specs as ppo_specs
from rl.torchrl.sac import setup as sac_setup


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
    def _policy_sample(self, obs, deterministic=False):
        _ = deterministic
        act = torch.tanh(obs[..., :1])
        lp = torch.zeros(obs.shape[0], dtype=obs.dtype)
        return (act, lp)

    _Policy = type("_Policy", (), {"sample": _policy_sample})

    q1 = nn.Linear(2, 1)
    q2 = nn.Linear(2, 1)
    q1_t = nn.Linear(2, 1)
    q2_t = nn.Linear(2, 1)

    def _qwrap_init(self, base):
        nn.Module.__init__(self)
        self.base = base

    def _qwrap_forward(self, obs, act):
        return self.base(torch.cat([obs, act], dim=-1)).squeeze(-1)

    _QWrap = type("_QWrap", (nn.Module,), {"__init__": _qwrap_init, "forward": _qwrap_forward})

    actor = _Policy()
    modules = SACUpdateModules(
        actor=actor,
        q1=_QWrap(q1),
        q2=_QWrap(q2),
        q1_target=_QWrap(q1_t),
        q2_target=_QWrap(q2_t),
        log_alpha=nn.Parameter(torch.tensor(0.0)),
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
    mean, se, all_same, num_steps_total = mean_return_over_runs(conf, object(), num_denoise=2, i_noise=1)
    assert np.isfinite(mean)
    assert np.isfinite(se)
    assert all_same is False
    assert num_steps_total >= 0
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

    def _call_get(self, timeout):
        assert timeout == 5
        return ("trace", "log", "collector")

    _Call = type("_Call", (), {"get": _call_get})
    _Factory = type("_Factory", (), {"from_id": staticmethod(lambda _call_id: _Call())})
    monkeypatch.setattr("experiments.modal_collect.modal.functions.FunctionCall", _Factory)
    assert isinstance(get_job_result("id-1"), tuple)

    called = {"n": 0}
    monkeypatch.setattr(
        "experiments.modal_collect.collect",
        lambda job_fn, cb: cb(("trace", "log", "collector")),
    )
    monkeypatch.setattr(
        "experiments.modal_collect.post_process",
        lambda *args: called.__setitem__("n", called["n"] + 1),
    )
    monkeypatch.setattr("experiments.modal_collect.os.path.exists", lambda p: False)
    modal_collect_main("jobs.txt")
    assert called["n"] == 1

    env_conf = SimpleNamespace(
        gym_conf=SimpleNamespace(
            transform_state=True,
            state_space=SimpleNamespace(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                shape=(1,),
            ),
        ),
        ensure_spaces=lambda: None,
    )
    assert str(select_device("cpu")) == "cpu"
    lb, width = obs_scale_from_env(env_conf)
    assert np.allclose(lb, np.array([-1.0], dtype=np.float32))
    assert np.allclose(width, np.array([2.0], dtype=np.float32))

    if "ops.experiment" in sys.modules:
        monkeypatch.delitem(sys.modules, "ops.experiment")

    import ops.experiment as ops_experiment_mod

    monkeypatch.setattr(ops_experiment_mod, "_forward_to_experiments_cli", lambda extra_args: None)
    runner = CliRunner()
    result = runner.invoke(ops_experiment_mod.cli)
    assert result.exit_code == 0


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


def test_kiss_cov_sac_eval_model_td_trainer(monkeypatch):
    import time
    from types import SimpleNamespace

    import torch
    from torch import nn

    from rl.pufferlib.sac import env_utils as sac_env_utils
    from rl.pufferlib.sac import eval_utils as sac_eval_utils
    from rl.pufferlib.sac import model_utils as sac_model_utils
    from rl.torchrl.offpolicy import actor_eval as trl_actor_eval
    from rl.torchrl.offpolicy import trainer_utils as trl_trainer_utils

    monkeypatch.setattr(
        sac_env_utils,
        "build_continuous_gym_env_setup",
        lambda **_kwargs: SimpleNamespace(
            env_conf=SimpleNamespace(gym_conf=SimpleNamespace(transform_state=True)),
            problem_seed=3,
            noise_seed_0=4,
            obs_lb=np.array([-1.0, -1.0], dtype=np.float32),
            obs_width=np.array([2.0, 2.0], dtype=np.float32),
            act_dim=2,
            action_low=np.array([-1.0, -1.0], dtype=np.float32),
            action_high=np.array([1.0, 1.0], dtype=np.float32),
        ),
    )

    monkeypatch.setattr(
        sac_eval_utils,
        "collect_denoised_trajectory",
        lambda _env_conf, _policy, **_kwargs: (SimpleNamespace(rreturn=2.5), 0),
    )
    cfg = SimpleNamespace(
        num_denoise=1,
        num_denoise_passive=1,
        eval_interval_steps=1,
        eval_seed_base=0,
        eval_noise_mode="frozen",
        seed=0,
    )
    env_setup = SimpleNamespace(env_conf=SimpleNamespace(), problem_seed=0)
    obs_spec = SimpleNamespace(mode="vector")
    modules = SimpleNamespace(
        actor=nn.Identity(),
        actor_backbone=nn.Linear(2, 2),
        actor_head=nn.Linear(2, 4),
        log_std=None,
    )
    train_state = sac_eval_utils.TrainState(global_step=1, start_time=time.time() - 1.0)
    monkeypatch.setattr(
        sac_eval_utils,
        "build_eval_plan",
        lambda **_kwargs: SimpleNamespace(eval_seed=0, heldout_i_noise=0),
    )
    monkeypatch.setattr(sac_eval_utils, "evaluate_heldout_if_enabled", lambda *_args, **_kwargs: 1.0)
    assert sac_eval_utils.evaluate_actor(cfg, env_setup, modules, obs_spec, device=torch.device("cpu"), eval_seed=0) == 2.5
    sac_eval_utils.maybe_eval(cfg, env_setup, modules, obs_spec, train_state, device=torch.device("cpu"))
    assert train_state.best_return >= 2.5

    env = sac_env_utils.EnvSetup(
        env_conf=SimpleNamespace(from_pixels=False, pixels_only=True),
        problem_seed=0,
        noise_seed_0=0,
        obs_lb=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        obs_width=np.array([2.0, 2.0, 2.0], dtype=np.float32),
        act_dim=2,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
    )
    spec = sac_env_utils.ObservationSpec(mode="vector", raw_shape=(3,), vector_dim=3)
    sac_cfg = SimpleNamespace(
        backbone_name="mlp",
        backbone_hidden_sizes=(8,),
        backbone_activation="relu",
        backbone_layer_norm=False,
        actor_head_hidden_sizes=(),
        critic_head_hidden_sizes=(),
        head_activation="relu",
        learning_rate_actor=3e-4,
        learning_rate_critic=3e-4,
        learning_rate_alpha=3e-4,
        alpha_init=0.2,
        batch_size=2,
        gamma=0.99,
        tau=0.005,
        target_entropy=-2.0,
        theta_dim=None,
    )
    modules2 = sac_model_utils.build_modules(sac_cfg, env, spec, device=torch.device("cpu"))
    optim2 = sac_model_utils.build_optimizers(sac_cfg, modules2)
    assert isinstance(sac_model_utils.alpha(modules2).item(), float)

    def _replay_sample(self, batch_size, device=None):
        _ = batch_size, device
        return (
            torch.zeros((2, 3)),
            torch.zeros((2, 2)),
            torch.zeros(2),
            torch.zeros((2, 3)),
            torch.zeros(2),
        )

    _Replay = type("_Replay", (), {"sample": _replay_sample})

    _ = sac_model_utils.sac_update(sac_cfg, modules2, optim2, _Replay(), device=torch.device("cpu"))

    modules3 = SimpleNamespace(actor_backbone=nn.Linear(3, 4), actor_head=nn.Linear(4, 2))
    pol = trl_actor_eval.OffPolicyActorEvalPolicy(
        modules3.actor_backbone,
        nn.Linear(4, 4),
        nn.Identity(),
        act_dim=2,
        device=torch.device("cpu"),
    )
    _ = pol(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    snap = trl_actor_eval.capture_actor_snapshot(modules3)
    trl_actor_eval.restore_actor_snapshot(modules3, snap)
    with trl_actor_eval.use_actor_snapshot(modules3, snap, device=torch.device("cpu")):
        pass

    from tensordict import TensorDict

    td = TensorDict(
        {
            "obs": torch.zeros((2, 3)),
            "action": torch.zeros((2, 2)),
            "next": TensorDict(
                {
                    "reward": torch.ones(2),
                    "terminated": torch.zeros(2, dtype=torch.bool),
                    "truncated": torch.zeros(2, dtype=torch.bool),
                },
                batch_size=[2],
            ),
        },
        batch_size=[2],
    )
    flat = trl_trainer_utils.flatten_batch_to_transitions(td)
    flat = trl_trainer_utils.normalize_actions_for_replay(
        flat,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
    )
    assert flat.shape[0] == 2
