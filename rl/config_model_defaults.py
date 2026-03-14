from __future__ import annotations

from typing import Any

from rl.model_defaults import resolve_rl_model_defaults

_HIDDEN_SIZE_KEYS = (
    "backbone_hidden_sizes",
    "critic_backbone_hidden_sizes",
    "actor_head_hidden_sizes",
    "critic_head_hidden_sizes",
)


def _as_tuple_ints(value: Any, *, key: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple((int(x) for x in value))
    raise TypeError(f"'{key}' must be a list or tuple of ints.")


def _require_env_tag(data: dict[str, Any], *, algo: str) -> str:
    env_tag = data.get("env_tag")
    if env_tag is None or str(env_tag).strip() == "":
        raise ValueError(f"{algo.upper()} config must set a non-empty 'env_tag'.")
    return str(env_tag)


def _apply_env_model_defaults(
    raw: dict[str, Any],
    *,
    algo: str,
) -> dict[str, Any]:
    data = dict(raw)
    if algo == "ppo" and "value_head_hidden_sizes" in data:
        raise ValueError("PPO config uses canonical key 'critic_head_hidden_sizes' (not 'value_head_hidden_sizes').")
    env_tag = _require_env_tag(data, algo=algo)
    defaults = resolve_rl_model_defaults(env_tag, algo=algo)
    for key, value in defaults.items():
        data.setdefault(key, value)
    for key in _HIDDEN_SIZE_KEYS:
        if key in data and data[key] is not None:
            data[key] = _as_tuple_ints(data[key], key=key)
    return data


def apply_ppo_env_model_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    return _apply_env_model_defaults(raw, algo="ppo")


def apply_sac_env_model_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    return _apply_env_model_defaults(raw, algo="sac")
