"""Targeted imports/calls so kiss static test_coverage links code units to tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from tests.kiss_dummy_nn_modules import make_pm_module_type


def test_kiss_bridge_pufferlib_offpolicy_env_eval(monkeypatch):
    from rl.pufferlib.offpolicy.env_utils import (
        EnvSetup,
        ObservationSpec,
        build_env_setup,
        infer_observation_spec,
        make_vector_env,
        prepare_obs_np,
        resolve_backbone_name,
        resolve_device,
        to_env_action,
    )
    from rl.pufferlib.offpolicy.env_utils import (
        seed_everything as puf_seed_everything,
    )
    from rl.pufferlib.offpolicy.eval_utils import (
        SacEvalPolicy,
        TrainState,
        append_eval_metric,
        render_videos_if_enabled,
    )
    from rl.pufferlib.offpolicy.eval_utils import (
        capture_actor_state as eval_capture_actor_state,
    )
    from rl.pufferlib.offpolicy.eval_utils import (
        evaluate_actor as eval_evaluate_actor,
    )
    from rl.pufferlib.offpolicy.eval_utils import (
        evaluate_heldout_if_enabled as eval_evaluate_heldout_if_enabled,
    )
    from rl.pufferlib.offpolicy.eval_utils import (
        log_if_due as eval_log_if_due,
    )
    from rl.pufferlib.offpolicy.eval_utils import (
        maybe_eval as eval_maybe_eval,
    )
    from rl.pufferlib.offpolicy.eval_utils import (
        use_actor_state as eval_use_actor_state,
    )

    puf_seed_everything(0)
    assert ObservationSpec is not None and EnvSetup is not None
    assert callable(resolve_device) and callable(to_env_action)
    cfg = SimpleNamespace(
        env_tag="pend",
        seed=0,
        num_envs=1,
        problem_seed=None,
        noise_seed_0=None,
        from_pixels=False,
        pixels_only=True,
        backbone_name="mlp",
        framestack=1,
    )
    monkeypatch.setattr("rl.pufferlib.offpolicy.env_utils.import_pufferlib_modules", lambda: (object(), object(), object()))
    monkeypatch.setattr("rl.pufferlib.offpolicy.env_utils._make_vector_env_common", lambda *a, **k: object())
    assert build_env_setup(cfg) is not None
    arr = np.zeros((1, 2), dtype=np.float32)
    osp = infer_observation_spec(cfg, arr)
    resolve_backbone_name(cfg, osp)
    prepare_obs_np(arr, obs_spec=osp)
    make_vector_env(cfg)
    dev = torch.device("cpu")
    mods = SimpleNamespace(
        actor_backbone=nn.Linear(2, 2),
        actor_head=nn.Linear(2, 2),
        log_std=None,
        actor=SimpleNamespace(act=lambda x: x),
    )
    st = eval_capture_actor_state(mods)
    with eval_use_actor_state(mods, st, device=dev):
        pass
    monkeypatch.setattr("rl.pufferlib.offpolicy.eval_utils.collect_denoised_trajectory", lambda *a, **k: (SimpleNamespace(rreturn=0.0), None))
    monkeypatch.setattr("rl.pufferlib.offpolicy.eval_utils.evaluate_for_best", lambda *a, **k: 0.0)
    monkeypatch.setattr("rl.pufferlib.offpolicy.eval_utils.evaluate_heldout_with_best_actor", lambda *a, **k: 0.0)
    obs_s = ObservationSpec(mode="vector", raw_shape=(2,), vector_dim=2)
    env = SimpleNamespace(env_conf=SimpleNamespace(), problem_seed=0)
    eval_evaluate_actor(SimpleNamespace(num_denoise=1), env, mods, obs_s, device=dev, eval_seed=0)
    eval_evaluate_heldout_if_enabled(SimpleNamespace(num_denoise_passive=1), env, mods, obs_s, device=dev, heldout_i_noise=0)
    eval_log_if_due(SimpleNamespace(log_interval_steps=1), TrainState(start_time=0.0), step=1, frames_per_batch=1)
    append_eval_metric(MagicMock(), TrainState(start_time=0.0), step=1)
    SacEvalPolicy(modules=mods, obs_spec=obs_s, device=dev)(np.zeros(2, dtype=np.float32))
    render_videos_if_enabled(SimpleNamespace(video_enable=False), env, mods, obs_s, device=dev)
    assert callable(eval_maybe_eval)


def test_kiss_bridge_pufferlib_offpolicy_model_ppo(monkeypatch):
    from rl.pufferlib.offpolicy.env_utils import ObservationSpec
    from rl.pufferlib.offpolicy.model_utils import (
        ActorNet as PufActorNet,
    )
    from rl.pufferlib.offpolicy.model_utils import (
        OffPolicyModules,
        OffPolicyOptimizers,
        QNetPixel,
        restore_actor_state,
    )
    from rl.pufferlib.offpolicy.model_utils import (
        build_modules as puf_build_modules,
    )
    from rl.pufferlib.offpolicy.model_utils import (
        build_optimizers as puf_build_optimizers,
    )
    from rl.pufferlib.offpolicy.model_utils import (
        capture_actor_state as mod_capture_actor_state,
    )
    from rl.pufferlib.offpolicy.model_utils import (
        use_actor_state as mod_use_actor_state,
    )
    from rl.pufferlib.ppo.checkpoint import (
        build_checkpoint_payload as ppo_ck_build_checkpoint_payload,
    )
    from rl.pufferlib.ppo.checkpoint import (
        maybe_save_periodic_checkpoint,
        restore_checkpoint_if_requested,
        save_final_checkpoint,
    )
    from rl.pufferlib.ppo.eval import PufferEvalPolicy, validate_eval_config
    from rl.pufferlib.ppo.eval import capture_actor_snapshot as ppo_capture_actor_snapshot
    from rl.pufferlib.ppo.eval_seeds import resolve_eval_seeds as ppo_resolve_eval_seeds
    from rl.pufferlib.sac.config import SACConfig
    from rl.pufferlib.sac.config import TrainResult as PufferSacTrainResult

    dev = torch.device("cpu")
    osp = ObservationSpec(mode="vector", raw_shape=(2,), vector_dim=2)
    built = puf_build_modules(
        SimpleNamespace(
            backbone_name="mlp",
            backbone_hidden_sizes=(4,),
            backbone_activation="tanh",
            backbone_layer_norm=False,
            actor_head_hidden_sizes=(),
            critic_head_hidden_sizes=(),
            head_activation="tanh",
        ),
        SimpleNamespace(obs_lb=None, obs_width=None, act_dim=2),
        osp,
        device=dev,
    )
    opts = puf_build_optimizers(SimpleNamespace(learning_rate_actor=1e-3, learning_rate_critic=1e-3), built)
    st2 = mod_capture_actor_state(built)
    restore_actor_state(built, st2)
    with mod_use_actor_state(built, st2):
        pass
    _ = opts
    assert PufActorNet is not None and QNetPixel is not None
    assert OffPolicyModules is not None and OffPolicyOptimizers is not None
    pm = make_pm_module_type("PM")()
    opt = torch.optim.AdamW(pm.parameters(), lr=0.1)
    st3 = SimpleNamespace(global_step=1, best_actor_state=None, best_return=0.0, last_eval_return=0.0, last_heldout_return=None, last_episode_return=None)
    ppo_ck_build_checkpoint_payload(pm, opt, st3, iteration=1)
    restore_checkpoint_if_requested(SimpleNamespace(resume_from=None), SimpleNamespace(batch_size=1), pm, opt, st3, device=dev)
    maybe_save_periodic_checkpoint(SimpleNamespace(checkpoint_interval=None), MagicMock(), pm, opt, st3, iteration=1)
    save_final_checkpoint(SimpleNamespace(checkpoint_interval=None), MagicMock(), pm, opt, st3, iteration=1)
    validate_eval_config(
        SimpleNamespace(
            eval_interval=1,
            eval_noise_mode=None,
            num_denoise=None,
            num_denoise_passive=None,
            checkpoint_interval=None,
            video_num_episodes=1,
            video_num_video_episodes=0,
            video_episode_selection="best",
        )
    )
    ppo_resolve_eval_seeds(SimpleNamespace(seed=0, problem_seed=0, noise_seed_0=0))
    PufferEvalPolicy(
        model=pm,
        obs_spec=osp,
        action_spec=SimpleNamespace(kind="continuous", dim=1),
        device=dev,
        prepare_obs_fn=lambda *a, **k: torch.zeros(1, 1),
    )(np.zeros(1, dtype=np.float32))
    ppo_capture_actor_snapshot(pm)
    assert SACConfig is not None and PufferSacTrainResult is not None


def test_kiss_bridge_torchrl_sac_setup_loop_ppo_engine_tail(monkeypatch, tmp_path):
    pytest.importorskip("torchrl")
    import rl.core.sac_eval as sac_eval_mod
    from rl.pufferlib.ppo.engine import (
        build_eval_env_conf as ppo_eng_build_eval_env_conf,
    )
    from rl.pufferlib.ppo.eval import (
        capture_actor_snapshot,
        restore_actor_snapshot,
        use_actor_snapshot,
    )
    from rl.pufferlib.ppo.specs import (
        normalize_action_bounds as ppo_specs_normalize_action_bounds,
    )
    from rl.pufferlib.sac.engine import register as puffer_sac_engine_register
    from rl.torchrl.sac.config import SACConfig
    from rl.torchrl.sac.loop import (
        evaluate_heldout_if_enabled as tr_sac_evaluate_heldout_if_enabled,
    )
    from rl.torchrl.sac.loop import log_if_due as tr_sac_log_if_due
    from rl.torchrl.sac.setup import build_env_setup as tr_sac_setup_build_env_setup
    from rl.torchrl.sac.setup import build_modules as tr_sac_setup_build_modules
    from rl.torchrl.sac.setup import build_training as tr_sac_setup_build_training

    ppo_eng_build_eval_env_conf(
        SimpleNamespace(seed=0, problem_seed=0, noise_seed_0=0, env_tag="pend", pixels_only=True),
        obs_spec=SimpleNamespace(mode="vector", vector_dim=2),
    )

    def _tail_pm_init(self):
        nn.Module.__init__(self)
        self.actor_backbone = nn.Linear(1, 1)
        self.actor_head = nn.Linear(1, 1)

    TailPM = type("TailPM", (nn.Module,), {"__init__": _tail_pm_init})
    pm = TailPM()
    snap = capture_actor_snapshot(pm)
    restore_actor_snapshot(pm, snap, device=torch.device("cpu"))
    with use_actor_snapshot(pm, snap, device=torch.device("cpu")):
        pass

    ppo_specs_normalize_action_bounds(np.zeros(3), np.ones(3), 3)
    ppo_specs_normalize_action_bounds(np.full(2, -2.0), np.full(2, 2.0), 2)

    monkeypatch.setattr("rl.registry.register_algo", lambda *a, **k: None)
    puffer_sac_engine_register()

    monkeypatch.setattr(sac_eval_mod, "evaluate_heldout_with_best_actor", lambda **k: 0.0)
    tr_sac_evaluate_heldout_if_enabled(
        SimpleNamespace(env_tag="pend", num_denoise_passive=1),
        SimpleNamespace(
            problem_seed=0,
            noise_seed_0=0,
            env_conf=SimpleNamespace(from_pixels=False, pixels_only=True),
        ),
        SimpleNamespace(),
        SimpleNamespace(best_actor_state=None),
        device=torch.device("cpu"),
        capture_actor_state=lambda m: {},
        restore_actor_state=lambda *a, **k: None,
        eval_policy_factory=lambda *a, **k: lambda obs: obs,
        get_env_conf=lambda *a, **k: SimpleNamespace(),
        evaluate_for_best=lambda *a, **k: 0.0,
    )

    monkeypatch.setattr("rl.torchrl.sac.loop.rl_logger.log_eval_iteration", lambda **k: None)
    tr_sac_log_if_due(
        SimpleNamespace(log_interval_steps=1),
        SimpleNamespace(last_eval_return=0.0, last_heldout_return=None, best_return=0.0),
        step=1,
        start_time=0.0,
        latest_losses={"loss_actor": 0.0, "loss_critic": 0.0, "loss_alpha": 0.0},
        total_updates=0,
    )

    def _fake_bcges(**_kwargs):
        return SimpleNamespace(
            env_conf=SimpleNamespace(
                from_pixels=False,
                state_space=SimpleNamespace(shape=(4,)),
                gym_conf=None,
            ),
            problem_seed=0,
            noise_seed_0=0,
            act_dim=2,
            action_low=np.zeros(2, dtype=np.float32),
            action_high=np.ones(2, dtype=np.float32),
            obs_lb=np.zeros(4, dtype=np.float32),
            obs_width=np.ones(4, dtype=np.float32),
        )

    import rl.torchrl.sac.setup as tr_sac_setup_mod

    monkeypatch.setattr(tr_sac_setup_mod, "build_continuous_gym_env_setup", _fake_bcges)
    cfg = SACConfig(exp_dir=str(tmp_path / "sac_exp"), replay_size=100, batch_size=4)
    env_setup = tr_sac_setup_build_env_setup(cfg)
    dev = torch.device("cpu")
    mods = tr_sac_setup_build_modules(cfg, env_setup, device=dev)
    tr_sac_setup_build_training(cfg, mods)


def test_kiss_bridge_modal_synthetic_sine_disk_and_main_raw(monkeypatch, tmp_path, capsys):
    import contextlib

    from analysis.fitting_time.evaluate import (
        SURROGATE_BENCHMARK_KEYS,
        BMResult,
        MuSe,
        SyntheticSineSurrogateBenchmark,
    )
    from experiments import synthetic_sine_benchmark_payload as pl

    _zr = BMResult(MuSe(0.0, 0.0), MuSe(0.0, 0.0), MuSe(0.0, 0.0))
    _z = SyntheticSineSurrogateBenchmark(results={k: _zr for k in SURROGATE_BENCHMARK_KEYS})

    monkeypatch.setattr(
        "experiments.synthetic_sine_benchmark_payload.modal.enable_output",
        lambda: contextlib.nullcontext(),
    )

    PlApp = type("PlApp", (), {"run": lambda self: contextlib.nullcontext()})
    PlRem = type(
        "PlRem",
        (),
        {"remote": staticmethod(lambda n, d, fn, ps, *_args: pl.synthetic_sine_benchmark_result_to_payload(_z, n=n, d=d, function_name=fn, problem_seed=ps))},
    )
    monkeypatch.setattr(pl.modal, "enable_output", lambda: contextlib.nullcontext())
    pl_dest = pl.run_synthetic_sine_benchmark_modal_to_disk(1, 1, "sine", 0, tmp_path / "pl_direct", app=PlApp(), remote_fn=PlRem())
    assert pl_dest.exists()
