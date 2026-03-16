from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from rl.ppo.checkpoint import (
    build_checkpoint_payload,
    maybe_save_periodic_checkpoint,
    restore_checkpoint_if_requested,
    save_final_checkpoint,
)
from rl.ppo.eval import (
    capture_actor_snapshot,
    heldout,
    maybe_eval,
    render,
    restore_actor_snapshot,
    use_actor_snapshot,
)
from rl.ppo.eval_config import eval_conf


class _Model:
    def __init__(self):
        self.actor_backbone = nn.Linear(3, 4)
        self.actor_head = nn.Linear(4, 2)
        self.critic_backbone = nn.Linear(3, 4)
        self.critic_head = nn.Linear(4, 1)
        self.log_std = nn.Parameter(torch.zeros(2))


def test_direct_rl_coverage(monkeypatch, tmp_path: Path):
    ready = SimpleNamespace(gym_conf=SimpleNamespace(max_steps=7), env_name="ready", kwargs={})
    monkeypatch.setattr(
        "rl.ppo.eval_config.conf_for_run",
        lambda **_kw: SimpleNamespace(env_conf=ready, problem_seed=30, noise_seed_0=31),
    )
    monkeypatch.setattr("rl.ppo.eval_config.get_env_conf_fn", lambda: (lambda *_a, **_k: None))
    assert (
        eval_conf(
            SimpleNamespace(env_tag="cheetah", seed=1, problem_seed=None, noise_seed_0=None),
            obs_mode="vector",
            resolve_gym_env_name_fn=lambda _t: ("unused", {}),
        )
        is ready
    )

    model = _Model()
    snap = capture_actor_snapshot(model)
    with torch.no_grad():
        model.actor_head.weight.zero_()
    restore_actor_snapshot(model, snap, device=torch.device("cpu"))
    with use_actor_snapshot(model, snap, device=torch.device("cpu")):
        pass

    state = SimpleNamespace(
        global_step=16,
        start_iteration=0,
        best_actor_state=snap,
        best_return=1.0,
        last_eval_return=0.0,
        last_heldout_return=None,
        last_episode_return=2.0,
        eval_env_conf=ready,
        obs_spec="obs",
        action_spec=SimpleNamespace(kind="discrete"),
    )
    opt = torch.optim.Adam(nn.Linear(2, 2).parameters(), lr=1e-3)
    payload = build_checkpoint_payload(model, opt, state, iteration=2)
    assert payload["iteration"] == 2
    mgr = SimpleNamespace(
        calls=[],
        save_both=lambda payload, iteration: mgr.calls.append((payload, iteration)),
    )
    maybe_save_periodic_checkpoint(SimpleNamespace(checkpoint_interval=2), mgr, model, opt, state, iteration=1)
    maybe_save_periodic_checkpoint(SimpleNamespace(checkpoint_interval=2), mgr, model, opt, state, iteration=2)
    save_final_checkpoint(SimpleNamespace(checkpoint_interval=1), mgr, model, opt, state, iteration=3)
    assert len(mgr.calls) == 2
    monkeypatch.setattr(
        "rl.ppo.checkpoint.load_checkpoint",
        lambda _path, device: {
            "actor_backbone": model.actor_backbone.state_dict(),
            "actor_head": model.actor_head.state_dict(),
            "log_std": torch.zeros(2),
            "critic_backbone": model.critic_backbone.state_dict(),
            "critic_head": model.critic_head.state_dict(),
            "optimizer": opt.state_dict(),
            "iteration": 3,
            "global_step": 24,
            "best_actor_state": {"best": 9},
            "best_return": 6.0,
            "last_eval_return": 5.0,
            "last_heldout_return": 4.0,
            "last_episode_return": 8.0,
        },
    )
    restore_checkpoint_if_requested(
        SimpleNamespace(resume_from=str(tmp_path / "ckpt.pt")),
        SimpleNamespace(batch_size=8),
        model,
        opt,
        state,
        device=torch.device("cpu"),
    )
    assert state.start_iteration == 3

    monkeypatch.setattr("rl.ppo.eval.rl_eval.heldout", lambda **_kw: 6.0)
    assert (
        heldout(
            SimpleNamespace(num_denoise_passive=1),
            model,
            state,
            device=torch.device("cpu"),
            heldout_i_noise=9,
            env_fn=lambda *_a, **_k: ready,
            prepare_obs_fn=lambda obs, **_k: torch.as_tensor(obs, dtype=torch.float32),
        )
        == 6.0
    )

    monkeypatch.setattr("rl.ppo.eval.is_due", lambda *_a, **_k: True)
    monkeypatch.setattr(
        "rl.ppo.eval.rl_eval.plan",
        lambda **_kw: SimpleNamespace(eval_seed=7, heldout_i_noise=9),
    )
    monkeypatch.setattr("rl.ppo.eval.score", lambda *_a, **_k: 5.0)
    monkeypatch.setattr("rl.ppo.eval.rl_eval.update_best", lambda **_kw: (5.0, snap, True))
    monkeypatch.setattr("rl.ppo.eval.heldout", lambda *_a, **_k: 6.0)
    maybe_eval(
        SimpleNamespace(
            seed=3,
            problem_seed=None,
            eval_interval=1,
            eval_seed_base=None,
            eval_noise_mode=None,
            num_denoise=None,
            num_denoise_passive=1,
        ),
        model,
        state,
        iteration=1,
        device=torch.device("cpu"),
        env_fn=lambda *_a, **_k: ready,
        prepare_obs_fn=lambda obs, **_k: torch.as_tensor(obs, dtype=torch.float32),
    )

    calls = []
    monkeypatch.setattr(
        "rl.ppo.eval.common_video.render_policy_videos",
        lambda *_a, **kw: calls.append(kw),
    )
    render(
        SimpleNamespace(
            video_enable=True,
            problem_seed=None,
            seed=3,
            eval_seed_base=None,
            video_seed_base=None,
            video_prefix="p",
            video_num_episodes=2,
            video_num_video_episodes=1,
            video_episode_selection="best",
            env_tag="cheetah",
        ),
        model,
        state,
        exp_dir=tmp_path,
        device=torch.device("cpu"),
        env_fn=lambda *_a, **_k: ready,
        prepare_obs_fn=lambda obs, **_k: torch.as_tensor(obs, dtype=torch.float32),
    )
    assert calls
