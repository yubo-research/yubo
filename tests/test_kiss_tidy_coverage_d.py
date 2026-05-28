from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from kiss_rl_puffer_remaining_helpers import patch_torchrl_ppo_core_for_kiss

from rl.pufferlib.ppo.engine import train_ppo_puffer, train_ppo_puffer_impl
from rl.pufferlib.ppo.engine_impl import (
    config_proxy,
)
from rl.pufferlib.ppo.engine_impl import (
    register as ppo_engine_impl_register,
)
from rl.pufferlib.ppo.engine_impl import (
    train_ppo_puffer as engine_impl_train_ppo_puffer,
)
from rl.pufferlib.ppo.engine_impl import (
    train_ppo_puffer_impl as ppo_engine_impl_train_ppo_puffer_impl,
)
from rl.pufferlib.ppo.engine_impl_train_run import (
    train_ppo_puffer as train_run_train_ppo_puffer,
)
from rl.pufferlib.ppo.engine_impl_train_run import (
    train_ppo_puffer_impl as train_run_train_ppo_puffer_impl,
)
from rl.pufferlib.sac.engine import train_sac_puffer
from rl.pufferlib.sac.eval_utils_impl import (
    evaluate_actor,
    evaluate_heldout_if_enabled,
    maybe_eval,
)
from rl.pufferlib.sac.sac_puffer_engine_impl import (
    register as sac_puffer_engine_impl_register,
)
from rl.pufferlib.sac.sac_puffer_engine_impl import (
    train_sac_puffer as sac_puffer_engine_impl_train_sac_puffer,
)
from rl.pufferlib.sac.sac_puffer_engine_impl import (
    train_sac_puffer_impl as sac_puffer_engine_impl_train_sac_puffer_impl,
)
from rl.pufferlib.sac.sac_puffer_train_run import (
    train_sac_puffer as sac_puffer_train_run_train_sac_puffer,
)
from rl.pufferlib.sac.sac_puffer_train_run import (
    train_sac_puffer_impl as sac_puffer_train_run_train_sac_puffer_impl,
)
from rl.torchrl.offpolicy.trainer_utils import (
    flatten_batch_to_transitions,
    normalize_actions_for_replay,
)
from rl.torchrl.ppo.core_build import build_modules, build_training
from rl.torchrl.ppo.core_env_setup import build_env_setup
from tests.kiss_turbo_gp_helper import make_fake_gp
from turbo_m_ref.turbo_1_ask_tell_core import (
    create_candidates,
    init_counters_and_tr,
    init_hypers,
    sample_candidates,
    select_candidates,
    tell_impl,
    train_gp_model,
)
from turbo_m_ref.turbo_1_core import CandidatesResult


def _patch_ppo_train_noops(monkeypatch):
    monkeypatch.setattr("rl.pufferlib.ppo.training_ops.collect_rollout", lambda *a, **k: None)
    monkeypatch.setattr("rl.pufferlib.ppo.training_ops.compute_advantages", lambda *a, **k: (torch.zeros(1), torch.zeros(1)))
    monkeypatch.setattr("rl.pufferlib.ppo.training_ops.flatten_batch", lambda *a, **k: {})
    monkeypatch.setattr("rl.pufferlib.ppo.training_ops.ppo_update", lambda *a, **k: {})
    monkeypatch.setattr("rl.pufferlib.ppo.eval.maybe_eval_and_update_state", lambda *a, **k: None)
    monkeypatch.setattr("rl.pufferlib.ppo.eval.maybe_render_videos", lambda *a, **k: None)
    monkeypatch.setattr("rl.pufferlib.ppo.metrics._maybe_anneal_lr", lambda *a, **k: None)
    monkeypatch.setattr("rl.pufferlib.ppo.metrics._metric_payload", lambda *a, **k: {})
    monkeypatch.setattr("rl.pufferlib.ppo.metrics._append_metrics_line", lambda *a, **k: None)
    monkeypatch.setattr("rl.pufferlib.ppo.metrics._log_iteration", lambda *a, **k: None)
    monkeypatch.setattr("rl.pufferlib.ppo.checkpoint.restore_checkpoint_if_requested", lambda *a, **k: None)
    monkeypatch.setattr("rl.pufferlib.ppo.checkpoint.maybe_save_periodic_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr("rl.pufferlib.ppo.checkpoint.save_final_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr("rl.logger.log_run_header_basic", lambda *a, **k: None)
    monkeypatch.setattr("rl.logger.log_run_footer", lambda *a, **k: None)


def test_kiss_tidy_d_progress_policy_backbone(monkeypatch):
    from rl.backbone import BackboneSpec, HeadSpec
    from rl.core.progress import due_mark
    from rl.policy_backbone_actor import ActorBackbonePolicy, ActorBackbonePolicyFactory, ActorPolicySpec
    from rl.policy_backbone_atari import AtariMLP16DiscretePolicy
    from rl.policy_backbone_discrete import DiscreteActorBackbonePolicy, DiscreteActorBackbonePolicyFactory, DiscreteActorPolicySpec
    from rl.policy_backbone_gaussian import GaussianActorBackbonePolicy, GaussianActorBackbonePolicyFactory
    from rl.policy_backbone_utils import ensure_env_spaces, init_linear, obs_space_from_env_conf

    assert due_mark(10, 5, 0) == 2
    assert due_mark(4, 5, 1) is None
    init_linear(torch.nn.Linear(2, 3))
    ec = SimpleNamespace(state_space=None, gym_conf=None)

    def _boom():
        raise AssertionError

    ec.ensure_spaces = _boom
    with pytest.raises(AssertionError):
        ensure_env_spaces(ec)
    ec2 = SimpleNamespace(state_space=None, gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(4,))))
    ec2.ensure_spaces = lambda: None
    ensure_env_spaces(ec2)
    assert obs_space_from_env_conf(ec2).shape == (4,)
    ec3 = SimpleNamespace(state_space=SimpleNamespace(shape=(2,)), ensure_spaces=lambda: None)
    assert obs_space_from_env_conf(ec3).shape == (2,)
    cont_env = SimpleNamespace(
        problem_seed=0,
        gym_conf=None,
        state_space=SimpleNamespace(shape=(3,)),
        action_space=SimpleNamespace(shape=(2,)),
        ensure_spaces=lambda: None,
    )
    spec = ActorPolicySpec(
        backbone=BackboneSpec(name="mlp", hidden_sizes=(8,), activation="relu", layer_norm=False),
        head=HeadSpec(hidden_sizes=(), activation="relu"),
    )
    pol = ActorBackbonePolicy(cont_env, spec)
    pol(torch.zeros(1, 3))
    pol.set_params(np.zeros_like(pol._flat_params_init))
    fac_a = ActorBackbonePolicyFactory(
        BackboneSpec(name="mlp", hidden_sizes=(8,), activation="relu", layer_norm=False),
        HeadSpec(hidden_sizes=(), activation="relu"),
    )
    assert fac_a(cont_env) is not None
    monkeypatch.setattr(
        "rl.core.env_contract.resolve_env_io_contract",
        lambda env_conf, default_image_size=84: SimpleNamespace(
            observation=SimpleNamespace(mode="vector", vector_dim=4, raw_shape=(4,), model_channels=None, image_size=None),
            action=SimpleNamespace(kind="discrete", dim=3, low=np.zeros(1, dtype=np.float32), high=np.ones(1, dtype=np.float32)),
        ),
    )
    d_env = SimpleNamespace(problem_seed=1, ensure_spaces=lambda: None)
    dspec = DiscreteActorPolicySpec(
        backbone=BackboneSpec(name="mlp", hidden_sizes=(8,), activation="relu", layer_norm=False),
        head=HeadSpec(hidden_sizes=(), activation="relu"),
    )
    DiscreteActorBackbonePolicy(d_env, dspec)(torch.zeros(4))
    dpf = DiscreteActorBackbonePolicyFactory(
        BackboneSpec(name="mlp", hidden_sizes=(8,), activation="relu", layer_norm=False),
        HeadSpec(hidden_sizes=(), activation="relu"),
    )
    assert dpf(d_env) is not None
    atari_env = SimpleNamespace(problem_seed=0, ensure_spaces=lambda: None)
    AtariMLP16DiscretePolicy(atari_env)(torch.zeros(4))
    g_env = SimpleNamespace(
        problem_seed=0,
        gym_conf=SimpleNamespace(),
        state_space=SimpleNamespace(shape=(3,)),
        action_space=SimpleNamespace(shape=(2,)),
        ensure_spaces=lambda: None,
    )
    GaussianActorBackbonePolicy(g_env, squash_mode="tanh_clip", deterministic_eval=True)(np.zeros(3, dtype=np.float32))
    assert GaussianActorBackbonePolicyFactory(squash_mode="clip")(g_env) is not None
    with pytest.raises(ValueError):
        GaussianActorBackbonePolicy(g_env, squash_mode="bad")


def test_kiss_tidy_d_puffer_ppo_engine(monkeypatch, tmp_path):
    from rl.pufferlib.ppo import engine_impl as ppo_engine_impl
    from rl.pufferlib.ppo.config import PufferPPOConfig, TrainResult
    from rl.pufferlib.ppo.engine_helpers import build_eval_env_conf, make_vector_env

    _real_run_training = importlib.import_module("rl.pufferlib.ppo.engine_train").run_training
    cfg = PufferPPOConfig(
        exp_dir=str(tmp_path),
        env_tag="pend",
        total_timesteps=8,
        num_envs=2,
        num_steps=2,
        num_minibatches=1,
        backbone_name="mlp",
        share_backbone=False,
    )
    fake_vec = SimpleNamespace(
        single_action_space=SimpleNamespace(shape=(2,), low=-np.ones(2, dtype=np.float32), high=np.ones(2, dtype=np.float32)),
        reset=lambda seed=None: (np.zeros((int(cfg.num_envs), 5), dtype=np.float32), {}),
        close=lambda: None,
    )
    monkeypatch.setattr("rl.pufferlib.ppo.engine_helpers._make_vector_env", lambda c: fake_vec)
    assert make_vector_env(cfg) is not None
    monkeypatch.setattr("rl.pufferlib.ppo.eval_config.build_eval_env_conf", lambda *a, **k: SimpleNamespace(ok=True))
    assert build_eval_env_conf(cfg, obs_spec=SimpleNamespace(mode="vector", vector_dim=3)).ok is True

    def _quick_train(*a, **k):
        return TrainResult(best_return=0.0, last_eval_return=0.0, last_heldout_return=None, num_iterations=1)

    monkeypatch.setattr("rl.pufferlib.ppo.engine_train.run_training", _quick_train)
    assert ppo_engine_impl_train_ppo_puffer_impl(cfg).num_iterations == 1
    assert train_ppo_puffer_impl(cfg).num_iterations == 1
    assert train_ppo_puffer(cfg).num_iterations == 1
    assert engine_impl_train_ppo_puffer(cfg).num_iterations == 1
    assert train_run_train_ppo_puffer_impl(cfg).num_iterations == 1
    assert train_run_train_ppo_puffer(cfg).num_iterations == 1
    monkeypatch.setattr("rl.pufferlib.ppo.engine_train.run_training", _real_run_training)
    monkeypatch.setattr("rl.registry.register_algo", lambda *a, **k: None)
    ppo_engine_impl_register()
    assert ppo_engine_impl.engine_helpers_proxy("_seed_everything") is not None
    assert config_proxy("PufferPPOConfig") is PufferPPOConfig
    assert ppo_engine_impl.engine_helpers_proxy("make_vector_env")(cfg) is not None


def test_kiss_tidy_d_puffer_run_training(monkeypatch, tmp_path):
    from rl.pufferlib.ppo import engine_train
    from rl.pufferlib.ppo.config import PufferPPOConfig, TrainResult

    cfg = PufferPPOConfig(
        exp_dir=str(tmp_path),
        env_tag="pend",
        total_timesteps=8,
        num_envs=2,
        num_steps=2,
        num_minibatches=1,
        backbone_name="mlp",
        share_backbone=False,
    )
    _patch_ppo_train_noops(monkeypatch)
    plan = SimpleNamespace(batch_size=4, num_iterations=1, num_envs=2, num_steps=2, minibatch_size=4)
    model = torch.nn.Linear(3, 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    st = SimpleNamespace(
        start_iteration=0,
        start_time=0.0,
        global_step=0,
        obs_spec=SimpleNamespace(mode="vector", vector_dim=3),
        action_spec=SimpleNamespace(dim=2),
        best_return=-float("inf"),
        last_eval_return=float("nan"),
        last_heldout_return=None,
        last_episode_return=float("nan"),
    )

    def _init_rt(config, pl, dev, envs):
        return (model, opt, (3,), SimpleNamespace(), st)

    res = engine_train.run_training(
        cfg,
        plan,
        torch.device("cpu"),
        Path(tmp_path) / "m.jsonl",
        SimpleNamespace(),
        build_eval_env_conf_fn=lambda c, obs_spec=None: SimpleNamespace(),
        init_runtime_fn=_init_rt,
        prepare_obs_fn=lambda o, obs_spec=None, device=None: torch.zeros(1),
    )
    assert isinstance(res, TrainResult)


def test_kiss_tidy_d_sac_eval_utils_impl(monkeypatch):
    _sac_ev = importlib.import_module("rl.pufferlib.sac.eval_utils")
    for _n in (
        "collect_denoised_trajectory",
        "evaluate_for_best",
        "build_eval_plan",
        "evaluate_heldout_if_enabled",
    ):
        _sac_ev.__dict__.pop(_n, None)
    from rl.pufferlib.sac import eval_utils_impl as sac_eval_impl

    device = torch.device("cpu")
    monkeypatch.setattr("rl.core.episode_rollout.collect_denoised_trajectory", lambda *a, **k: (SimpleNamespace(rreturn=0.5), None))
    monkeypatch.setattr("rl.core.episode_rollout.evaluate_for_best", lambda *a, **k: 0.25)
    monkeypatch.setattr("rl.pufferlib.offpolicy.eval_utils.evaluate_heldout_with_best_actor", lambda **k: 0.1)
    monkeypatch.setattr("rl.eval_noise.build_eval_plan", lambda **k: SimpleNamespace(eval_seed=1, heldout_i_noise=2))
    monkeypatch.setattr("rl.pufferlib.offpolicy.eval_utils.update_best_actor_if_improved", lambda **k: (0.5, {}, True))
    monkeypatch.setattr("rl.pufferlib.offpolicy.eval_utils.due_mark", lambda *a, **k: 1)
    monkeypatch.setattr("rl.pufferlib.offpolicy.eval_utils.capture_actor_state", lambda m: {"x": 1})
    monkeypatch.setattr("rl.pufferlib.offpolicy.eval_utils.use_actor_state", lambda m, s, device=None: __import__("contextlib").nullcontext())
    mods = SimpleNamespace()
    ev = SimpleNamespace(env_conf=SimpleNamespace())
    ob = SimpleNamespace(mode="vector")
    assert evaluate_actor(SimpleNamespace(num_denoise=1), ev, mods, ob, device=device, eval_seed=0) == 0.5
    assert (
        evaluate_heldout_if_enabled(
            SimpleNamespace(num_denoise_passive=1),
            ev,
            mods,
            ob,
            device=device,
            heldout_i_noise=0,
            best_actor_state=None,
        )
        == 0.25
    )
    assert (
        evaluate_heldout_if_enabled(
            SimpleNamespace(num_denoise_passive=1),
            ev,
            mods,
            ob,
            device=device,
            heldout_i_noise=0,
            best_actor_state={},
            with_actor_state_fn=lambda s: __import__("contextlib").nullcontext(),
        )
        == 0.1
    )
    assert evaluate_heldout_if_enabled(SimpleNamespace(num_denoise_passive=None), ev, mods, ob, device=device, heldout_i_noise=0) is None
    tr = SimpleNamespace(global_step=10, eval_mark=0, best_return=0.0, best_actor_state=None, last_eval_return=0.0, last_heldout_return=None)
    ev2 = SimpleNamespace(env_conf=SimpleNamespace(), problem_seed=7)
    maybe_eval(
        SimpleNamespace(
            eval_interval_steps=5,
            eval_seed_base=None,
            eval_noise_mode=None,
            seed=1,
            num_denoise=1,
            num_denoise_passive=1,
        ),
        ev2,
        mods,
        ob,
        state=tr,
        device=device,
    )
    assert tr.eval_mark == 1
    assert getattr(sac_eval_impl, "rl_logger") is not None


def test_kiss_tidy_d_turbo_refs(monkeypatch):
    import turbo_m_ref.turbo_1_ask_tell_core as atc
    import turbo_m_ref.turbo_1_core as t1c

    atc.validate_init_args(
        np.zeros(2),
        np.ones(2),
        n_init=2,
        batch_size=1,
        verbose=False,
        use_ard=False,
        max_cholesky_size=0,
        n_training_steps=30,
        device="cpu",
        dtype="float32",
    )
    assert atc.standardize_fX(np.array([1.0, 2.0, 9.0])).sigma > 0
    xc = np.random.default_rng(0).random((6, 2)).astype(np.float64)
    yc = np.random.default_rng(1).standard_normal((6, 2)).astype(np.float64)
    assert select_candidates(2, 2, xc, yc).shape == (2, 2)
    self_at = SimpleNamespace(
        dim=2,
        n_cand=20,
        min_cuda=10**9,
        device=torch.device("cpu"),
        dtype=torch.float64,
        max_cholesky_size=0,
        use_ard=False,
        batch_size=1,
    )

    gp = make_fake_gp()
    monkeypatch.setattr(atc, "train_gp_model", lambda self, X, fX, n_training_steps, hypers, device, dtype: gp)
    monkeypatch.setattr(atc, "sample_candidates", lambda gp, X_cand, device, dtype, batch_size, max_cholesky_size: np.zeros((len(X_cand), batch_size)))
    assert callable(train_gp_model)
    assert callable(sample_candidates)
    init_hypers(self_at)
    init_counters_and_tr(self_at, batch_size=1, length_fixed=False)
    atc.device_dtype_for(self_at, 1)
    atc.trust_region_bounds(self_at, np.random.random((4, 2)), np.random.random(4), gp, 0.5)
    assert create_candidates(self_at, np.random.random((4, 2)), np.random.random(4), 0.5, 30, {}).X_cand is not None
    assert callable(tell_impl)

    t1c.validate_init_args(
        np.zeros(2),
        np.ones(2),
        n_init=2,
        max_evals=100,
        batch_size=1,
        verbose=False,
        use_ard=False,
        max_cholesky_size=0,
        n_training_steps=30,
        dtype="float32",
    )
    self_c = SimpleNamespace(dim=2, use_ard=False, _surrogate_type="none", batch_size=1)
    CandidatesResult(X_cand=np.zeros((1, 2)), y_cand=None, hypers={})
    t1c.make_X_cand(self_at, x_center=np.zeros((1, 2)), lb=np.zeros(2), ub=np.ones(2), device=torch.device("cpu"), dtype=torch.float64)
    assert (
        t1c.compute_y_cand(
            self_c,
            X=np.zeros((3, 2)),
            fX=np.zeros((3, 1)),
            X_cand=np.zeros((2, 2)),
            mu=0.0,
            sigma=1.0,
            gp=None,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        is None
    )


def test_kiss_tidy_d_sac_puffer_torchrl_ppo_core(monkeypatch, tmp_path):
    from rl.pufferlib.sac.config import SACConfig
    from rl.pufferlib.sac.sac_puffer_train_run_orchestrate import train_sac_puffer_impl as orch_impl
    from rl.torchrl.ppo.actor_nets import ActorNet
    from rl.torchrl.ppo.config import PPOConfig
    from rl.torchrl.ppo.ppo_nets_base import _BackboneHeadNet

    sac_cfg = SACConfig(exp_dir=str(tmp_path), env_tag="pend", device="cpu", total_timesteps=0, replay_backend="auto")
    assert sac_puffer_train_run_train_sac_puffer_impl(sac_cfg).num_steps == 0
    assert sac_puffer_train_run_train_sac_puffer(sac_cfg).num_steps == 0
    assert sac_puffer_engine_impl_train_sac_puffer_impl(sac_cfg).num_steps == 0
    assert sac_puffer_engine_impl_train_sac_puffer(sac_cfg).num_steps == 0
    assert train_sac_puffer(sac_cfg).num_steps == 0
    assert orch_impl(sac_cfg).num_steps == 0
    monkeypatch.setattr("rl.registry.register_algo", lambda *a, **k: None)
    sac_puffer_engine_impl_register()
    assert sac_puffer_engine_impl_register is not None

    td = __import__("tensordict").TensorDict(
        {
            "observation": torch.zeros(2, 3),
            "action": torch.zeros(2, 1),
            "next": __import__("tensordict").TensorDict(
                {"observation": torch.zeros(2, 3), "reward": torch.zeros(2), "terminated": torch.zeros(2, dtype=torch.bool)},
                batch_size=[2],
            ),
        },
        batch_size=[2],
    )
    flat = flatten_batch_to_transitions(td)
    assert flat.batch_size[0] == 2
    normalize_actions_for_replay(flat, action_low=np.array([-1.0], dtype=np.float32), action_high=np.array([1.0], dtype=np.float32))

    patch_torchrl_ppo_core_for_kiss(monkeypatch)
    pcfg = PPOConfig(env_tag="x", exp_dir=str(tmp_path), total_timesteps=4, num_envs=1, num_steps=2, num_minibatches=1)
    env_setup = build_env_setup(pcfg)
    mods = build_modules(pcfg, env_setup, device=torch.device("cpu"))
    rt = SimpleNamespace(collector_backend="multi_sync", single_env_backend="serial", device=torch.device("cpu"), collector_workers=1)
    trn = build_training(pcfg, env_setup, mods, runtime=rt)
    assert trn.frames_per_batch == 2
    oc = __import__("rl.core.env_contract", fromlist=["ObservationContract"]).ObservationContract
    c = oc(mode="vector", raw_shape=(3,), vector_dim=3)
    log_std = torch.nn.Parameter(torch.zeros(2))
    an = ActorNet(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2), log_std, torch.nn.Identity(), obs_contract=c)
    loc, _ = an.forward(torch.zeros(1, 3))
    assert loc.shape[-1] == 2
    bhn = _BackboneHeadNet(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2), torch.nn.Identity(), obs_contract=c)
    assert bhn.backbone is not None
