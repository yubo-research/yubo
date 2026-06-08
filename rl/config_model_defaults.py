from __future__ import annotations

from types import SimpleNamespace
from typing import Any

_HIDDEN_SIZE_KEYS = (
    "backbone_hidden_sizes",
    "actor_head_hidden_sizes",
    "critic_head_hidden_sizes",
)
_MODEL_CONFIG_KEYS = {
    "backbone_name",
    "backbone_hidden_sizes",
    "backbone_activation",
    "backbone_layer_norm",
    "actor_head_hidden_sizes",
    "critic_head_hidden_sizes",
    "head_activation",
    "share_backbone",
    "log_std_init",
    "theta_dim",
    "value_head_hidden_sizes",
}
_MODEL_DEFAULTS = {
    "ppo": {
        "backbone_name": "mlp",
        "backbone_hidden_sizes": (64, 64),
        "backbone_activation": "silu",
        "backbone_layer_norm": True,
        "actor_head_hidden_sizes": (),
        "critic_head_hidden_sizes": (),
        "head_activation": "silu",
        "share_backbone": True,
        "log_std_init": 0.0,
    },
    "sac": {
        "backbone_name": "mlp",
        "backbone_hidden_sizes": (256, 256),
        "backbone_activation": "silu",
        "backbone_layer_norm": False,
        "actor_head_hidden_sizes": (),
        "critic_head_hidden_sizes": (),
        "head_activation": "silu",
    },
}


def _as_tuple_ints(value: Any, *, key: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return tuple((int(x) for x in value))
    if isinstance(value, list):
        return tuple((int(x) for x in value))
    raise TypeError(f"'{key}' must be a list or tuple of ints.")


def _require_env_tag(data: dict[str, Any], *, algo: str) -> str:
    env_tag = data.get("env_tag")
    if env_tag is None or str(env_tag).strip() == "":
        raise ValueError(f"{algo.upper()} config must set a non-empty 'env_tag'.")
    return str(env_tag)


def _maybe_register_atari_dm_backends(env_tag: str) -> None:
    if not str(env_tag).startswith(("atari:", "ALE/", "dm:", "dm_control/")):
        return
    _ns: dict = {}
    exec("from problems.env_conf_backends import register_with_env_conf", _ns)  # noqa: S102
    _ns["register_with_env_conf"]()


def _apply_env_model_defaults(
    raw: dict[str, Any],
    *,
    algo: str,
) -> dict[str, Any]:
    _ns: dict = {}
    exec("from problems.problem import resolve_rl_model_defaults", _ns)  # noqa: S102
    resolve_rl_model_defaults = _ns["resolve_rl_model_defaults"]

    data = dict(raw)
    if algo == "ppo" and "value_head_hidden_sizes" in data:
        raise ValueError("PPO config uses canonical key 'critic_head_hidden_sizes' (not 'value_head_hidden_sizes').")
    env_tag = _require_env_tag(data, algo=algo)
    _maybe_register_atari_dm_backends(env_tag)
    pt = data.get("policy_tag")
    policy_tag: str | None = None if pt is None else (str(pt).strip() or None)
    defaults = resolve_rl_model_defaults(env_tag, policy_tag, algo=algo)
    for key, value in defaults.items():
        data.setdefault(key, value)
    for key in _HIDDEN_SIZE_KEYS:
        if key in data and data[key] is not None:
            data[key] = _as_tuple_ints(data[key], key=key)
    return data


def reject_model_config_keys(raw: dict[str, Any], *, algo: str) -> None:
    present = sorted(k for k in raw if k in _MODEL_CONFIG_KEYS)
    if present:
        keys = ", ".join(present)
        raise ValueError(f"{algo.upper()} config should use policy_tag for model architecture; remove explicit model keys: {keys}.")


def _resolve_model_settings(config: Any, *, algo: str) -> SimpleNamespace:
    _ns: dict = {}
    exec("from problems.problem import resolve_rl_model_defaults", _ns)  # noqa: S102
    resolve_rl_model_defaults = _ns["resolve_rl_model_defaults"]

    algo_key = str(algo).strip().lower()
    settings = dict(_MODEL_DEFAULTS[algo_key])
    settings.update(
        resolve_rl_model_defaults(
            str(config.env_tag),
            getattr(config, "policy_tag", None),
            algo=algo_key,
        )
    )
    for key in _HIDDEN_SIZE_KEYS:
        if key in settings and settings[key] is not None:
            settings[key] = _as_tuple_ints(settings[key], key=key)
    return SimpleNamespace(**settings)


def resolve_ppo_model_settings(config: Any) -> SimpleNamespace:
    return _resolve_model_settings(config, algo="ppo")


def resolve_sac_model_settings(config: Any) -> SimpleNamespace:
    return _resolve_model_settings(config, algo="sac")


def apply_ppo_env_model_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    return _apply_env_model_defaults(raw, algo="ppo")


def apply_sac_env_model_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    return _apply_env_model_defaults(raw, algo="sac")
