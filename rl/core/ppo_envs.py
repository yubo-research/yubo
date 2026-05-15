from __future__ import annotations

from common.env_tags import is_atari_env_tag, parse_atari_tag
from problems.env_conf_backends import maybe_register_atari_dm_backends

__all__ = ["is_atari_env_tag", "parse_atari_tag", "resolve_gym_env_name", "to_puffer_game_name"]


def _maybe_register_atari_dm_backends(env_tag: str) -> None:
    maybe_register_atari_dm_backends(str(env_tag))


def to_puffer_game_name(env_tag: str) -> str:
    ale_id = parse_atari_tag(str(env_tag))
    game = ale_id.split("/", 1)[1]
    return game.split("-v", 1)[0].lower()


def resolve_gym_env_name(env_tag: str) -> tuple[str, dict]:
    from problems.problem import build_problem

    maybe_register_atari_dm_backends(str(env_tag))
    # policy_tag="linear" is a placeholder; only problem.env is used (policy is never built)
    problem = build_problem(str(env_tag), policy_tag="linear")
    env = problem.env
    if str(getattr(env, "env_name", "")).startswith("dm_control/"):
        return (str(env.env_name), dict(getattr(env, "kwargs", {}) or {}))
    if getattr(env, "gym_conf", None) is not None:
        return (str(env.env_name), dict(getattr(env, "kwargs", {}) or {}))
    tag = str(env_tag)
    if ":" not in tag and "/" not in tag and ("-v" in tag):
        return (tag, {})
    raise ValueError(f"Unsupported non-Atari env tag for puffer backends: {env_tag}")
