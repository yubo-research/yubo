from __future__ import annotations

from rl.env_provider import get_env_conf_fn


def is_atari_env_tag(env_tag: str) -> bool:
    return str(env_tag).startswith(("atari:", "ALE/"))


def to_puffer_game_name(env_tag: str) -> str:
    tag = str(env_tag)
    if tag.startswith("atari:"):
        return tag.split(":", 1)[1].split(":", 1)[0].split("-")[0].lower()
    if tag.startswith("ALE/"):
        return tag.split("/", 1)[1].split("-v", 1)[0].lower()
    raise ValueError(f"Expected atari:Game or ALE/Game-v5, got: {env_tag}")


def resolve_gym_env_name(env_tag: str) -> tuple[str, dict]:
    env_conf = get_env_conf_fn()(str(env_tag))
    if str(getattr(env_conf, "env_name", "")).startswith("dm_control/"):
        return (str(env_conf.env_name), dict(getattr(env_conf, "kwargs", {}) or {}))
    if getattr(env_conf, "gym_conf", None) is not None:
        return (str(env_conf.env_name), dict(getattr(env_conf, "kwargs", {}) or {}))
    tag = str(env_tag)
    if ":" not in tag and "/" not in tag and ("-v" in tag):
        return (tag, {})
    raise ValueError(f"Unsupported non-Atari env tag for puffer backends: {env_tag}")
