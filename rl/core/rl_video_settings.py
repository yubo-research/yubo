from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RLVideoSettings:
    enable: bool = False
    prefix: str = "policy"
    num_episodes: int = 10
    num_video_episodes: int = 3
    episode_selection: str = "best"
    seed_base: int | None = None


_VIDEO_KEY_MAP = {
    "video_enable": "enable",
    "video_prefix": "prefix",
    "video_num_episodes": "num_episodes",
    "video_num_video_episodes": "num_video_episodes",
    "video_episode_selection": "episode_selection",
    "video_seed_base": "seed_base",
}


def pop_video_settings(data: dict[str, Any]) -> RLVideoSettings:
    values = {}
    for source_key, target_key in _VIDEO_KEY_MAP.items():
        if source_key in data:
            values[target_key] = data.pop(source_key)
    return RLVideoSettings(**values)


def attach_video_settings(config: Any, settings: RLVideoSettings) -> Any:
    setattr(config, "_video_settings", settings)
    return config


def get_video_settings(config: Any) -> RLVideoSettings:
    settings = getattr(config, "_video_settings", None)
    if isinstance(settings, RLVideoSettings):
        return settings
    values = {}
    for source_key, target_key in _VIDEO_KEY_MAP.items():
        if hasattr(config, source_key):
            values[target_key] = getattr(config, source_key)
    return RLVideoSettings(**values)
