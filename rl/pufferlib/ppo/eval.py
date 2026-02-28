from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def resolve_eval_seeds(config) -> tuple[int, int]:
    from rl.seed_util import resolve_noise_seed_0, resolve_problem_seed

    problem_seed = resolve_problem_seed(seed=int(config.seed), problem_seed=config.problem_seed)
    noise_seed_0 = resolve_noise_seed_0(problem_seed=problem_seed, noise_seed_0=config.noise_seed_0)
    return int(problem_seed), int(noise_seed_0)


class PufferEvalPolicy:
    def __init__(
        self,
        *,
        model,
        obs_spec,
        action_spec,
        device: torch.device,
        prepare_obs_fn,
    ):
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
    snapshot = {
        "backbone": {name: tensor.detach().clone() for name, tensor in model.actor_backbone.state_dict().items()},
        "head": {name: tensor.detach().clone() for name, tensor in model.actor_head.state_dict().items()},
    }
    if hasattr(model, "log_std"):
        snapshot["log_std"] = model.log_std.detach().cpu().clone()
    return snapshot


def restore_actor_snapshot(model, snapshot: dict, *, device: torch.device) -> None:
    model.actor_backbone.load_state_dict(snapshot["backbone"])
    model.actor_head.load_state_dict(snapshot["head"])
    if hasattr(model, "log_std") and "log_std" in snapshot:
        model.log_std.data.copy_(torch.as_tensor(snapshot["log_std"], device=device))


@contextmanager
def use_actor_snapshot(model, snapshot: dict, *, device: torch.device):
    previous_snapshot = capture_actor_snapshot(model)
    restore_actor_snapshot(model, snapshot, device=device)
    try:
        yield
    finally:
        restore_actor_snapshot(model, previous_snapshot, device=device)


def _get_eval_env_conf(config, state, *, build_eval_env_conf_fn):
    if state.eval_env_conf is None:
        state.eval_env_conf = build_eval_env_conf_fn(config, obs_spec=state.obs_spec)
    return state.eval_env_conf


def _evaluate_actor(
    config,
    model,
    state,
    *,
    device: torch.device,
    eval_seed: int,
    build_eval_env_conf_fn,
    prepare_obs_fn,
) -> float:
    opt_traj = __import__("optimizer.opt_trajectories", fromlist=["collect_denoised_trajectory"])
    eval_env_conf = _get_eval_env_conf(config, state, build_eval_env_conf_fn=build_eval_env_conf_fn)
    eval_policy = PufferEvalPolicy(
        model=model,
        obs_spec=state.obs_spec,
        action_spec=state.action_spec,
        device=device,
        prepare_obs_fn=prepare_obs_fn,
    )
    traj, _ = opt_traj.collect_denoised_trajectory(
        eval_env_conf,
        eval_policy,
        num_denoise=config.num_denoise_eval,
        i_noise=int(eval_seed),
    )
    return float(traj.rreturn)


def _evaluate_heldout_if_enabled(
    config,
    model,
    state,
    *,
    device: torch.device,
    heldout_i_noise: int,
    build_eval_env_conf_fn,
    prepare_obs_fn,
) -> float | None:
    if config.num_denoise_passive_eval is None or state.best_actor_state is None:
        return None
    opt_traj = __import__("optimizer.opt_trajectories", fromlist=["evaluate_for_best"])
    eval_env_conf = _get_eval_env_conf(config, state, build_eval_env_conf_fn=build_eval_env_conf_fn)
    eval_policy = PufferEvalPolicy(
        model=model,
        obs_spec=state.obs_spec,
        action_spec=state.action_spec,
        device=device,
        prepare_obs_fn=prepare_obs_fn,
    )
    with use_actor_snapshot(model, state.best_actor_state, device=device):
        return float(
            opt_traj.evaluate_for_best(
                eval_env_conf,
                eval_policy,
                config.num_denoise_passive_eval,
                i_noise=int(heldout_i_noise),
            )
        )


def maybe_eval_and_update_state(
    config,
    model,
    state,
    *,
    iteration: int,
    device: torch.device,
    build_eval_env_conf_fn,
    prepare_obs_fn,
) -> None:
    interval = int(config.eval_interval)
    if interval <= 0 or iteration % interval != 0:
        return

    eval_noise = __import__("rl.eval_noise", fromlist=["build_eval_plan"])
    plan = eval_noise.build_eval_plan(
        current=iteration,
        interval=interval,
        seed=int(config.seed),
        eval_seed_base=config.eval_seed_base,
        eval_noise_mode=config.eval_noise_mode,
    )
    state.last_eval_return = _evaluate_actor(
        config,
        model,
        state,
        device=device,
        eval_seed=plan.eval_seed,
        build_eval_env_conf_fn=build_eval_env_conf_fn,
        prepare_obs_fn=prepare_obs_fn,
    )
    if state.last_eval_return > state.best_return:
        state.best_return = float(state.last_eval_return)
        state.best_actor_state = capture_actor_snapshot(model)
    state.last_heldout_return = _evaluate_heldout_if_enabled(
        config,
        model,
        state,
        device=device,
        heldout_i_noise=plan.heldout_i_noise,
        build_eval_env_conf_fn=build_eval_env_conf_fn,
        prepare_obs_fn=prepare_obs_fn,
    )


def maybe_render_videos(
    config,
    model,
    state,
    *,
    exp_dir: Path,
    device: torch.device,
    build_eval_env_conf_fn,
    prepare_obs_fn,
) -> None:
    if not bool(config.video_enable):
        return
    eval_env_conf = _get_eval_env_conf(config, state, build_eval_env_conf_fn=build_eval_env_conf_fn)
    if getattr(eval_env_conf, "gym_conf", None) is None:
        print(
            f"video disabled for non-gym env: {config.env_tag}",
            flush=True,
        )
        return

    video_dir = exp_dir / "videos"

    base_seed = int(
        config.video_seed_base if config.video_seed_base is not None else (config.eval_seed_base if config.eval_seed_base is not None else config.seed)
    )
    actor_state = state.best_actor_state if state.best_actor_state is not None else capture_actor_snapshot(model)
    eval_policy = PufferEvalPolicy(
        model=model,
        obs_spec=state.obs_spec,
        action_spec=state.action_spec,
        device=device,
        prepare_obs_fn=prepare_obs_fn,
    )
    video = __import__("common.video", fromlist=["render_policy_videos"])
    with use_actor_snapshot(model, actor_state, device=device):
        video.render_policy_videos(
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
        eval_noise = __import__("rl.eval_noise", fromlist=["normalize_eval_noise_mode"])
        eval_noise.normalize_eval_noise_mode(config.eval_noise_mode)
    if config.num_denoise_eval is not None and int(config.num_denoise_eval) <= 0:
        raise ValueError("num_denoise_eval must be > 0 when provided.")
    if config.num_denoise_passive_eval is not None and int(config.num_denoise_passive_eval) <= 0:
        raise ValueError("num_denoise_passive_eval must be > 0 when provided.")
    if config.checkpoint_interval is not None and int(config.checkpoint_interval) <= 0:
        raise ValueError("checkpoint_interval must be > 0 when provided.")
    if int(config.video_num_episodes) <= 0:
        raise ValueError("video_num_episodes must be > 0.")
    if int(config.video_num_video_episodes) < 0:
        raise ValueError("video_num_video_episodes must be >= 0.")
    selection = str(config.video_episode_selection).lower()
    if selection not in ("best", "first", "random"):
        raise ValueError("video_episode_selection must be one of: best, first, random.")
