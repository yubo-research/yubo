from __future__ import annotations


def validate_eggroll_jax_objective_env(env_name: str, error_cls: type[Exception]) -> None:
    from problems.isaaclab_env_adapters import is_isaaclab_env_tag
    from problems.jax_env_core import supports_jax_objective_tag

    if is_isaaclab_env_tag(env_name):
        return
    if not supports_jax_objective_tag(env_name):
        raise error_cls(f"EggRoll requires a supported JAX objective env tag (got {env_name!r}).")
