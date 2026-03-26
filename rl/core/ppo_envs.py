from __future__ import annotations


def _maybe_register_atari_dm_backends(env_tag: str) -> None:
    if not str(env_tag).startswith(("atari:", "ALE/", "dm:", "dm_control/")):
        return
    _ns: dict = {}
    exec("from problems.env_conf_backends import register_with_env_conf", _ns)  # noqa: S102
    _ns["register_with_env_conf"]()


def _env_tag_for_problem_build(env_tag: str, *, from_pixels: bool) -> str:
    t = str(env_tag)
    if from_pixels and (t.startswith("dm:") or t.startswith("dm_control/")):
        parts = t.split(":")
        if parts[-1] != "pixels":
            return f"{t}:pixels"
    return t


def is_atari_env_tag(env_tag: str) -> bool:
    return str(env_tag).startswith(("atari:", "ALE/"))


def to_puffer_game_name(env_tag: str) -> str:
    _ns: dict = {}
    exec("from problems.atari_env import _parse_atari_tag", _ns)  # noqa: S102
    ale_id = _ns["_parse_atari_tag"](str(env_tag))
    game = ale_id.split("/", 1)[1]
    return game.split("-v", 1)[0].lower()


def resolve_gym_env_name(env_tag: str) -> tuple[str, dict]:
    from problems.problem import build_problem

    _maybe_register_atari_dm_backends(str(env_tag))
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
