#!/usr/bin/env python3

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def main() -> None:
    from problems.isaaclab_env_adapters import isaaclab_video_launcher_kwargs, make_isaaclab_env
    from video.isaaclab_viewport import prepare_isaaclab_video_view

    env = make_isaaclab_env(
        "isaaclab:Isaac-Velocity-Flat-G1-v0",
        seed=0,
        num_envs=1,
        launcher_kwargs=isaaclab_video_launcher_kwargs(),
    )
    try:
        env.reset(seed=0)
        for _ in range(8):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        raw_env = getattr(env, "_env", env)
        from problems.isaaclab_env_adapters import get_isaaclab_session

        prepare_isaaclab_video_view(get_isaaclab_session().app, raw_env)
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("no active viewport")
        out = Path("isaaclab_viewport_probe.png")
        capture_viewport_to_file(viewport, str(out))
        for _ in range(80):
            get_isaaclab_session().app.update()
            if out.exists() and out.stat().st_size > 0:
                print("VIEWPORT_CAPTURE_OK", out, out.stat().st_size, flush=True)
                return
            time.sleep(0.05)
        raise RuntimeError("viewport capture did not produce a file")
    finally:
        env.close()


if __name__ == "__main__":
    main()
