from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from common import video as common_video
from rl import eval_noise
from rl.core import ppo_eval
from rl.core.actor_state import capture_backbone_head_snapshot, restore_backbone_head_snapshot, use_backbone_head_snapshot
from rl.core.env_conf import resolve_run_seeds
from rl.core.episode_rollout import collect_denoised_trajectory, evaluate_for_best
from rl.core.progress import is_due


def resolve_eval_seeds(config) -> tuple[int, int]:
    resolved = resolve_run_seeds(seed=int(config.seed), problem_seed=config.problem_seed, noise_seed_0=config.noise_seed_0)
    return (int(resolved.problem_seed), int(resolved.noise_seed_0))


class PufferEvalPolicy:
    def __init__(self, *, model, obs_spec, action_spec, device: torch.device, prepare_obs_fn):
        self._model = model
        self._obs_spec = obs_spec
        self._action_spec = action_spec
        self._device = device
        self._prepare_obs_fn = prepare_obs_fn

    def __call__(self, state: np.ndarray) -> np.ndarray:
        state_np = np.asarray(state)
        state_batch = np.expand_dims(state_np, axis=0)
        obs_t = self._prepare_obs_fn(state_batch, obs_spec=self._obs_spec, device=self._device)
        with torch.no_grad():
            actor_out = self._model.actor_head(self._model.actor_backbone(obs_t))
            if self._action_spec.kind == "discrete":
                action_t = actor_out.argmax(dim=-1).float()
            else:
                action_t = torch.tanh(actor_out)
        return action_t.squeeze(0).detach().cpu().numpy()


def capture_actor_snapshot(model) -> dict:
    return capture_backbone_head_snapshot(
        model.actor_backbone, model.actor_head, log_std=getattr(model, "log_std", None), state_to_cpu=False, log_std_to_cpu=True, log_std_format="tensor"
    )


def restore_actor_snapshot(model, snapshot: dict, *, device: torch.device) -> None:
    restore_backbone_head_snapshot(model.actor_backbone, model.actor_head, snapshot, log_std=getattr(model, "log_std", None), device=device)


@contextmanager
def use_actor_snapshot(model, snapshot: dict, *, device: torch.device):
    with use_backbone_head_snapshot(
        model.actor_backbone,
        model.actor_head,
        snapshot,
        log_std=getattr(model, "log_std", None),
        device=device,
        state_to_cpu=False,
        log_std_to_cpu=True,
        log_std_format="tensor",
    ):
        yield


def _get_eval_env_conf(config, state, *, build_eval_env_conf_fn):
    if state.eval_env_conf is None:
        state.eval_env_conf = build_eval_env_conf_fn(config, obs_spec=state.obs_spec)
    return state.eval_env_conf


def _evaluate_actor(config, model, state, *, device: torch.device, eval_seed: int, build_eval_env_conf_fn, prepare_obs_fn) -> float:
    eval_env_conf = _get_eval_env_conf(config, state, build_eval_env_conf_fn=build_eval_env_conf_fn)
    eval_policy = PufferEvalPolicy(model=model, obs_spec=state.obs_spec, action_spec=state.action_spec, device=device, prepare_obs_fn=prepare_obs_fn)
    traj, _ = collect_denoised_trajectory(eval_env_conf, eval_policy, num_denoise=config.num_denoise, i_noise=int(eval_seed))
    return float(traj.rreturn)


def _evaluate_heldout_if_enabled(config, model, state, *, device: torch.device, heldout_i_noise: int, build_eval_env_conf_fn, prepare_obs_fn) -> float | None:
    eval_env_conf = _get_eval_env_conf(config, state, build_eval_env_conf_fn=build_eval_env_conf_fn)
    eval_policy = PufferEvalPolicy(model=model, obs_spec=state.obs_spec, action_spec=state.action_spec, device=device, prepare_obs_fn=prepare_obs_fn)
    return ppo_eval.evaluate_heldout_with_best_actor(
        best_actor_state=state.best_actor_state,
        num_denoise_passive=config.num_denoise_passive,
        heldout_i_noise=int(heldout_i_noise),
        with_actor_state=lambda snapshot: use_actor_snapshot(model, snapshot, device=device),
        evaluate_for_best=evaluate_for_best,
        eval_env_conf=eval_env_conf,
        eval_policy=eval_policy,
    )


def maybe_eval_and_update_state(config, model, state, *, iteration: int, device: torch.device, build_eval_env_conf_fn, prepare_obs_fn) -> None:
    interval = int(config.eval_interval)
    if not is_due(int(iteration), interval):
        return
    run_problem_seed = int(config.problem_seed) if config.problem_seed is not None else int(config.seed)
    plan = eval_noise.build_eval_plan(
        current=iteration, interval=interval, seed=run_problem_seed, eval_seed_base=config.eval_seed_base, eval_noise_mode=config.eval_noise_mode
    )
    state.last_eval_return = _evaluate_actor(
        config, model, state, device=device, eval_seed=plan.eval_seed, build_eval_env_conf_fn=build_eval_env_conf_fn, prepare_obs_fn=prepare_obs_fn
    )
    state.best_return, state.best_actor_state, _ = ppo_eval.update_best_actor_if_improved(
        eval_return=float(state.last_eval_return),
        best_return=float(state.best_return),
        best_actor_state=state.best_actor_state,
        capture_actor_state=lambda: capture_actor_snapshot(model),
    )
    state.last_heldout_return = _evaluate_heldout_if_enabled(
        config, model, state, device=device, heldout_i_noise=plan.heldout_i_noise, build_eval_env_conf_fn=build_eval_env_conf_fn, prepare_obs_fn=prepare_obs_fn
    )


def maybe_render_videos(config, model, state, *, exp_dir: Path, device: torch.device, build_eval_env_conf_fn, prepare_obs_fn) -> None:
    if not bool(config.video_enable):
        return
    eval_env_conf = _get_eval_env_conf(config, state, build_eval_env_conf_fn=build_eval_env_conf_fn)
    if getattr(eval_env_conf, "gym_conf", None) is None:
        print(f"video disabled for non-gym env: {config.env_tag}", flush=True)
        return
    video_dir = exp_dir / "videos"
    run_problem_seed = int(config.problem_seed) if config.problem_seed is not None else int(config.seed)
    base_seed = int(
        config.video_seed_base if config.video_seed_base is not None else config.eval_seed_base if config.eval_seed_base is not None else run_problem_seed
    )
    actor_state = state.best_actor_state if state.best_actor_state is not None else capture_actor_snapshot(model)
    eval_policy = PufferEvalPolicy(model=model, obs_spec=state.obs_spec, action_spec=state.action_spec, device=device, prepare_obs_fn=prepare_obs_fn)
    with use_actor_snapshot(model, actor_state, device=device):
        common_video.render_policy_videos(
            eval_env_conf,
            eval_policy,
            video_dir=video_dir,
            video_prefix=str(config.video_prefix),
            num_episodes=int(config.video_num_episodes),
            num_video_episodes=int(config.video_num_video_episodes),
            episode_selection=str(config.video_episode_selection or "best"),
            seed_base=base_seed,
        )


def validate_eval_config(config) -> None:
    interval = int(config.eval_interval)
    if interval < 0:
        raise ValueError("eval_interval must be >= 0.")
    if config.eval_noise_mode is not None:
        eval_noise.normalize_eval_noise_mode(config.eval_noise_mode)
    if config.num_denoise is not None and int(config.num_denoise) <= 0:
        raise ValueError("num_denoise must be > 0 when provided.")
    if config.num_denoise_passive is not None and int(config.num_denoise_passive) <= 0:
        raise ValueError("num_denoise_passive must be > 0 when provided.")
    if config.checkpoint_interval is not None and int(config.checkpoint_interval) <= 0:
        raise ValueError("checkpoint_interval must be > 0 when provided.")
    if int(config.video_num_episodes) <= 0:
        raise ValueError("video_num_episodes must be > 0.")
    if int(config.video_num_video_episodes) < 0:
        raise ValueError("video_num_video_episodes must be >= 0.")
    selection = str(config.video_episode_selection).lower()
    if selection not in ("best", "first", "random"):
        raise ValueError("video_episode_selection must be one of: best, first, random.")
