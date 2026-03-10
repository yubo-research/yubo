from __future__ import annotations


def _maybe_register_atari_dm_backends(env_tag: str) -> None:
    if not str(env_tag).startswith(("atari:", "ALE/", "dm:", "dm_control/")):
        return
    from problems.env_conf_backends import register_with_env_conf

    register_with_env_conf()


def is_atari_env_tag(env_tag: str) -> bool:
    return str(env_tag).startswith(("atari:", "ALE/"))


def to_puffer_game_name(env_tag: str) -> str:
    from problems.atari_env import _parse_atari_tag

    ale_id = _parse_atari_tag(str(env_tag))
    game = ale_id.split("/", 1)[1]
    return game.split("-v", 1)[0].lower()


def resolve_gym_env_name(env_tag: str) -> tuple[str, dict]:
    from problems.env_conf import get_env_conf

    _maybe_register_atari_dm_backends(str(env_tag))
    env_conf = get_env_conf(str(env_tag))
    if str(getattr(env_conf, "env_name", "")).startswith("dm_control/"):
        return (str(env_conf.env_name), dict(getattr(env_conf, "kwargs", {}) or {}))
    if getattr(env_conf, "gym_conf", None) is not None:
        return (str(env_conf.env_name), dict(getattr(env_conf, "kwargs", {}) or {}))
    tag = str(env_tag)
    if ":" not in tag and "/" not in tag and ("-v" in tag):
        return (tag, {})
    raise ValueError(f"Unsupported non-Atari env tag for puffer backends: {env_tag}")
