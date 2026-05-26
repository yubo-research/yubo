from __future__ import annotations

from typing import Any

from problems.isaaclab_config import disable_command_debug_visualizers
from video.isaaclab_viewport import prepare_isaaclab_video_view


def make_raw_isaaclab_env(
    adapters: Any,
    env_tag: str,
    *,
    headless: bool = True,
    num_envs: int = 1,
    device: str | None = None,
    render_mode: str | None = None,
    launcher_kwargs: dict[str, Any] | None = None,
    **kwargs,
):
    task_id = adapters.parse_isaaclab_task_id(env_tag)
    if launcher_kwargs is None and render_mode == "rgb_array":
        launcher_kwargs = adapters.isaaclab_video_launcher_kwargs()
    session = adapters.get_isaaclab_session(headless=headless, launcher_kwargs=launcher_kwargs)

    make_kwargs = dict(kwargs)
    make_kwargs.pop("batched", None)
    seed = make_kwargs.pop("seed", None)
    with adapters._capture_isaaclab_setup_output():
        if "cfg" not in make_kwargs:
            make_kwargs["cfg"] = adapters._parse_env_cfg(task_id, num_envs=int(num_envs), device=device)
        if seed is not None and hasattr(make_kwargs["cfg"], "seed"):
            try:
                make_kwargs["cfg"].seed = int(seed)
            except Exception:
                pass
        if render_mode is not None:
            make_kwargs["render_mode"] = render_mode

        disable_command_debug_visualizers(make_kwargs["cfg"])
        env = session.gym.make(task_id, **make_kwargs)
        if render_mode == "rgb_array":
            prepare_isaaclab_video_view(session.app, env)
    return env
