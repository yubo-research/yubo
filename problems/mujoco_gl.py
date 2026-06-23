from __future__ import annotations

import os
import sys

_DARWIN_INVALID_GL_BACKENDS = {"egl", "glx", "osmesa"}


def normalize_mujoco_gl_for_platform() -> None:
    """Keep MuJoCo imports from seeing a backend invalid for this platform."""
    mujoco_gl = os.environ.get("MUJOCO_GL", "").lower().strip()
    if sys.platform == "darwin" and mujoco_gl in _DARWIN_INVALID_GL_BACKENDS:
        os.environ["MUJOCO_GL"] = "cgl"
