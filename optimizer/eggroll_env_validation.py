from __future__ import annotations


def validate_eggroll_jax_objective_env(env_name: str, error_cls: type[Exception]) -> None:
    from problems.jax_env_core import supports_jax_objective_tag

    if not supports_jax_objective_tag(env_name):
        raise error_cls(f"EggRoll requires a supported JAX objective env tag (got {env_name!r}).")
