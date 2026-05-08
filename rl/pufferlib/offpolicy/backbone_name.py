from __future__ import annotations

from typing import Any


def resolve_backbone_name(config: Any, obs_spec: Any) -> str:
    if getattr(obs_spec, "mode", "vector") != "pixels":
        return str(config.backbone_name)
    channels = int(getattr(obs_spec, "channels", 3) or 3)
    key = str(config.backbone_name).strip().lower()
    if key in {"mlp", "nature_cnn"} and channels == 4:
        return "nature_cnn_atari"
    if key in {"mlp", "nature_cnn_atari"} and channels != 4:
        return "nature_cnn"
    return str(config.backbone_name)
