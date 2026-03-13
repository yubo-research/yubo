from __future__ import annotations

OBS_MODES = frozenset({"vector", "image", "mixed"})


def normalize_obs_mode(obs_mode: str | None) -> str:
    mode = "vector" if obs_mode is None else str(obs_mode).strip().lower()
    if mode not in OBS_MODES:
        raise ValueError(f"obs_mode must be one of: {sorted(OBS_MODES)} (got: {obs_mode})")
    return mode


def obs_mode_uses_pixels(obs_mode: str) -> bool:
    return normalize_obs_mode(obs_mode) in {"image", "mixed"}


def obs_mode_pixels_only(obs_mode: str) -> bool:
    return normalize_obs_mode(obs_mode) == "image"


def obs_mode_from_flags(*, from_pixels: bool, pixels_only: bool) -> str:
    if not bool(from_pixels):
        return "vector"
    return "image" if bool(pixels_only) else "mixed"
