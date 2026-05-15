import copy
from types import SimpleNamespace
from typing import Any

from policies.mlp_policy import MLPPolicyFactory
from problems.env_conf_bindings import get_atari_dm_bindings
from problems.env_conf_parse import parse_tag_options
from problems.env_conf_presets import (
    _atari_env_confs,
    _dm_control_env_confs,
    _gym_env_confs,
)
from problems.isaaclab_env_adapters import is_isaaclab_env_tag

_ISAACLAB_RL_MODEL = {
    "ppo": {
        "backbone_name": "mlp",
        "backbone_hidden_sizes": (64, 64),
        "backbone_activation": "silu",
        "backbone_layer_norm": True,
        "actor_head_hidden_sizes": (),
        "critic_head_hidden_sizes": (),
        "head_activation": "silu",
        "share_backbone": True,
        "log_std_init": -0.5,
    },
    "sac": {
        "backbone_name": "mlp",
        "backbone_hidden_sizes": (64, 64),
        "backbone_activation": "silu",
        "backbone_layer_norm": True,
        "actor_head_hidden_sizes": (),
        "critic_head_hidden_sizes": (),
        "head_activation": "silu",
    },
}


def _normalize_rl_env_key(env_tag: str) -> str:
    tag, _frozen_noise, _from_pixels = parse_tag_options(str(env_tag), None)
    if is_isaaclab_env_tag(tag):
        return tag
    if tag.startswith("dm:"):
        env_name, _policy_class = get_atari_dm_bindings().resolve_dm_control_from_tag(tag, False)
        return str(env_name)
    if tag.startswith("atari:"):
        env_id, _policy_class = get_atari_dm_bindings().resolve_atari_from_tag(tag)
        return str(env_id)
    return tag


def _find_rl_env_conf(env_key: str):
    if is_isaaclab_env_tag(env_key):
        return SimpleNamespace(rl_model=copy.deepcopy(_ISAACLAB_RL_MODEL), policy_class=None)
    registries = (_gym_env_confs, _dm_control_env_confs, _atari_env_confs)
    for registry in registries:
        direct = registry.get(env_key)
        if direct is not None:
            return direct
    for registry in registries:
        for conf in registry.values():
            if getattr(conf, "env_name", None) == env_key:
                return conf
    return None


def _infer_rl_from_policy_class(policy_class: Any, *, algo: str) -> dict[str, Any] | None:
    if isinstance(policy_class, MLPPolicyFactory):
        hidden = tuple((int(v) for v in policy_class._hidden_sizes))
        layer_norm = bool(policy_class._use_layer_norm)
        share_backbone = True
        log_std_init = -0.5
    elif hasattr(policy_class, "_hidden_sizes") and hasattr(policy_class, "_share_backbone") and hasattr(policy_class, "_log_std_init"):
        hidden = tuple((int(v) for v in policy_class._hidden_sizes))
        layer_norm = True
        share_backbone = bool(policy_class._share_backbone)
        log_std_init = float(policy_class._log_std_init)
    else:
        return None

    if algo == "ppo":
        return {
            "backbone_name": "mlp",
            "backbone_hidden_sizes": hidden,
            "backbone_activation": "silu",
            "backbone_layer_norm": layer_norm,
            "actor_head_hidden_sizes": (),
            "critic_head_hidden_sizes": (),
            "head_activation": "silu",
            "share_backbone": share_backbone,
            "log_std_init": log_std_init,
        }
    if algo == "sac":
        return {
            "backbone_name": "mlp",
            "backbone_hidden_sizes": hidden,
            "backbone_activation": "silu",
            "backbone_layer_norm": layer_norm,
            "actor_head_hidden_sizes": (),
            "critic_head_hidden_sizes": (),
            "head_activation": "silu",
        }
    raise ValueError(f"Unsupported algo '{algo}' for RL model inference.")


def _explicit_rl_model_for_algo(env_conf: Any, *, algo: str) -> dict[str, Any] | None:
    rl_model = getattr(env_conf, "rl_model", None)
    if not isinstance(rl_model, dict):
        return None
    model = rl_model.get(algo)
    if not isinstance(model, dict):
        return None
    return copy.deepcopy(model)


def _inferred_rl_model_for_algo(env_conf: Any, *, algo: str) -> dict[str, Any] | None:
    policy_class = getattr(env_conf, "policy_class", None)
    if policy_class is None:
        return None
    inferred = _infer_rl_from_policy_class(policy_class, algo=algo)
    if inferred is None:
        return None
    return copy.deepcopy(inferred)


def resolve_rl_model_defaults(env_tag: str, *, algo: str) -> dict[str, Any]:
    algo_key = str(algo).strip().lower()
    if algo_key not in {"ppo", "sac"}:
        raise ValueError(f"Unsupported algo '{algo}'. Expected one of: ppo, sac.")
    env_key = _normalize_rl_env_key(str(env_tag))
    env_conf = _find_rl_env_conf(env_key)
    if env_conf is None:
        raise ValueError(f"No env preset found for env_tag '{env_tag}'. Add an entry to one of: _gym_env_confs, _dm_control_env_confs, _atari_env_confs.")
    providers = (_explicit_rl_model_for_algo, _inferred_rl_model_for_algo)
    for provider in providers:
        model = provider(env_conf, algo=algo_key)
        if model is not None:
            return copy.deepcopy(model)
    raise ValueError(
        f"No RL model defaults for env_tag '{env_tag}' and algo '{algo_key}'. "
        "Provide env_conf.rl_model[algo] or use an inferable MLPPolicyFactory policy_class."
    )
