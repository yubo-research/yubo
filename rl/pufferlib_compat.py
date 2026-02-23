"""Compatibility helpers for using PufferLib in a Gymnasium-only environment."""

from __future__ import annotations

import sys
from types import ModuleType


def _install_gym_alias() -> ModuleType:
    # Enforce a single backend: any `import gym` resolves to Gymnasium.
    # This prevents drift when both gym and gymnasium are installed.
    import gym as gym_alias

    sys.modules["gym"] = gym_alias

    wrappers = getattr(gym_alias, "wrappers", None)
    if wrappers is not None and not hasattr(wrappers, "FrameStack") and hasattr(wrappers, "FrameStackObservation"):
        setattr(wrappers, "FrameStack", wrappers.FrameStackObservation)
    return gym_alias


def import_pufferlib_modules():
    _install_gym_alias()
    try:
        import pufferlib
        import pufferlib.environments.atari as puffer_atari
        import pufferlib.vector as puffer_vector
    except Exception as exc:  # pragma: no cover - exercised in integration envs
        raise ImportError(
            "Failed to import pufferlib backend. Install with `pip install pufferlib --no-deps` "
            "in the active environment and ensure ale-py ROM setup is complete."
        ) from exc

    return pufferlib, puffer_vector, puffer_atari
