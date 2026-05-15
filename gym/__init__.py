"""Compatibility shim for packages that import `gym`."""

from __future__ import annotations

import sys

import gymnasium as _gymnasium

wrappers = getattr(_gymnasium, "wrappers", None)
if wrappers is not None and not hasattr(wrappers, "FrameStack"):
    frame_stack_obs = getattr(wrappers, "FrameStackObservation", None)
    if frame_stack_obs is not None:
        setattr(wrappers, "FrameStack", frame_stack_obs)

sys.modules[__name__] = _gymnasium
