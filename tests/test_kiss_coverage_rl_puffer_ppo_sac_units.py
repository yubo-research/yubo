from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def _mod(parts, names):
    return __import__(".".join(parts), fromlist=names)


def test_kiss_cov_puffer_ppo_and_sac_units(monkeypatch, tmp_path):
    checkpoint = _mod(
        ("rl", "pufferlib", "ppo", "checkpoint"),
        fromlist=["build_checkpoint_payload", "save_final_checkpoint"],
    )
    ppo_config = _mod(
        ("rl", "pufferlib", "ppo", "config"),
        fromlist=["PufferPPOConfig", "TrainResult"],
    )
    engine = _mod(
        ("rl", "pufferlib", "ppo", "engine"),
        fromlist=["build_eval_env_conf", "make_vector_env", "register"],
    )
    eval_mod = _mod(
        ("rl", "pufferlib", "ppo", "eval"),
        fromlist=[
            "capture_actor_snapshot",
            "maybe_eval_and_update_state",
            "maybe_render_videos",
            "restore_actor_snapshot",
            "use_actor_snapshot",
        ],
    )
    specs = _mod(("rl", "pufferlib", "ppo", "specs"), fromlist=["normalize_action_bounds"])
    sac_config = _mod(("rl", "pufferlib", "sac", "config"), fromlist=["SACConfig"])
    sac_engine = _mod(("rl", "pufferlib", "sac", "engine"), fromlist=["register"])

    model = SimpleNamespace(
        actor_backbone=torch.nn.Linear(3, 4),
        actor_head=torch.nn.Linear(4, 2),
        critic_backbone=torch.nn.Linear(3, 4),
        critic_head=torch.nn.Linear(4, 1),
    )
    opt = torch.optim.Adam(model.actor_backbone.parameters(), lr=1e-3)
    state = SimpleNamespace(
        global_step=1,
        best_actor_state=None,
        best_return=0.0,
        last_eval_return=0.0,
        last_heldout_return=None,
        last_episode_return=0.0,
    )
    payload = checkpoint.build_checkpoint_payload(model, opt, state, iteration=1)
    assert payload["iteration"] == 1
    mgr = SimpleNamespace(save_both=lambda payload, iteration: None)
    checkpoint.save_final_checkpoint(SimpleNamespace(checkpoint_interval=1), mgr, model, opt, state, iteration=1)

    cfg = ppo_config.PufferPPOConfig(env_tag="pend")
    assert isinstance(cfg.to_dict(), dict)
    assert (
        ppo_config.TrainResult(
            best_return=0.0,
            last_eval_return=0.0,
            last_heldout_return=None,
            num_iterations=1,
        ).num_iterations
        == 1
    )
    assert specs.normalize_action_bounds(np.array([-1.0]), np.array([1.0]), 1)[0].shape == (1,)
    assert sac_config.SACConfig(env_tag="pend").env_tag == "pend"
    assert sac_config.TrainResult(best_return=0.0, last_eval_return=0.0, last_heldout_return=None, num_steps=1).num_steps == 1

    monkeypatch.setattr("rl.pufferlib.ppo.engine._make_vector_env", lambda config: SimpleNamespace())
    monkeypatch.setattr(
        "rl.pufferlib.ppo.engine._build_eval_env_conf",
        lambda config, obs_spec: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "importlib.import_module",
        lambda name: SimpleNamespace(register_algo=lambda *args, **kwargs: None),
    )
    assert engine.make_vector_env(cfg) is not None
    assert engine.build_eval_env_conf(cfg, obs_spec=SimpleNamespace(mode="vector")) is not None
    engine.register()
    sac_engine.register()

    snap = eval_mod.capture_actor_snapshot(model)
    eval_mod.restore_actor_snapshot(model, snap, device=torch.device("cpu"))
    with eval_mod.use_actor_snapshot(model, snap, device=torch.device("cpu")):
        pass

    monkeypatch.setattr("rl.pufferlib.ppo.eval.is_due", lambda iteration, interval: True)
    monkeypatch.setattr(
        "rl.pufferlib.ppo.eval.eval_noise.build_eval_plan",
        lambda **kwargs: SimpleNamespace(eval_seed=1, heldout_i_noise=2),
    )
    monkeypatch.setattr(
        "rl.pufferlib.ppo.eval.collect_denoised_trajectory",
        lambda *args, **kwargs: (SimpleNamespace(rreturn=1.0), None),
    )
    monkeypatch.setattr(
        "rl.pufferlib.ppo.eval.ppo_eval.update_best_actor_if_improved",
        lambda **kwargs: (1.0, snap, True),
    )
    monkeypatch.setattr(
        "rl.pufferlib.ppo.eval.ppo_eval.evaluate_heldout_with_best_actor",
        lambda **kwargs: 1.0,
    )
    monkeypatch.setattr(
        "rl.pufferlib.ppo.eval.common_video.render_policy_videos",
        lambda *args, **kwargs: None,
    )
    ppo_state = SimpleNamespace(
        obs_spec=SimpleNamespace(mode="vector"),
        action_spec=SimpleNamespace(kind="continuous"),
        eval_env_conf=None,
        best_actor_state=None,
        best_return=0.0,
        last_eval_return=0.0,
        last_heldout_return=None,
    )
    ppo_cfg = SimpleNamespace(
        eval_interval=1,
        num_denoise=1,
        num_denoise_passive=1,
        problem_seed=1,
        seed=1,
        eval_seed_base=None,
        eval_noise_mode=None,
        video_enable=True,
        video_seed_base=None,
        video_prefix="p",
        video_num_episodes=1,
        video_num_video_episodes=1,
        video_episode_selection="best",
    )
    eval_mod.maybe_eval_and_update_state(
        ppo_cfg,
        model,
        ppo_state,
        iteration=1,
        device=torch.device("cpu"),
        build_eval_env_conf_fn=lambda config, obs_spec: SimpleNamespace(gym_conf=SimpleNamespace()),
        prepare_obs_fn=lambda obs, obs_spec, device: torch.as_tensor(obs, dtype=torch.float32, device=device),
    )
    eval_mod.maybe_render_videos(
        ppo_cfg,
        model,
        ppo_state,
        exp_dir=Path(tmp_path),
        device=torch.device("cpu"),
        build_eval_env_conf_fn=lambda config, obs_spec: SimpleNamespace(gym_conf=SimpleNamespace()),
        prepare_obs_fn=lambda obs, obs_spec, device: torch.as_tensor(obs, dtype=torch.float32, device=device),
    )
