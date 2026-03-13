from __future__ import annotations

import copy
from typing import Any

from problems.rl_policy_factory import project
from rl.env_provider import get_env_conf_fn, register_get_env_conf


def resolve_rl_model_defaults(env_tag: str, *, algo: str) -> dict[str, Any]:
    algo_key = str(algo).strip().lower()
    if algo_key not in {"ppo", "sac"}:
        raise ValueError(f"Unsupported algo '{algo}'. Expected one of: ppo, sac.")
    get_env = get_env_conf_fn()
    if get_env is None:
        from problems.env_conf import get_env_conf

        register_get_env_conf(get_env_conf)
        get_env = get_env_conf
    env_conf = get_env(str(env_tag), problem_seed=0, noise_seed_0=0)
    policy_factory = getattr(env_conf, "policy_class", None)
    if policy_factory is None:
        raise ValueError(f"No RL model defaults for env_tag '{env_tag}' and algo '{algo_key}'. Provide a policy_class with an inferable RL schema.")
    to_rl_schema = getattr(policy_factory, "to_rl_schema", None)
    if to_rl_schema is None:
        raise ValueError(f"No RL model defaults for env_tag '{env_tag}' and algo '{algo_key}'. Provide a policy_class with an inferable RL schema.")
    return copy.deepcopy(project(dict(to_rl_schema()), algo_key))
