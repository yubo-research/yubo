"""Utilities for extracting observation and action spaces from env configurations."""

from typing import Any


def get_obs_space(env_conf: Any) -> Any:
    """Get observation space from env_conf, checking multiple locations.

    Args:
        env_conf: Environment configuration object

    Returns:
        Observation space object with .shape attribute

    Raises:
        ValueError: If observation space cannot be found
    """
    obs_space = getattr(env_conf, "state_space", None)
    if obs_space is None:
        gym_conf = getattr(env_conf, "gym_conf", None)
        obs_space = getattr(gym_conf, "state_space", None) if gym_conf is not None else None
    if obs_space is None:
        raise ValueError("Observation space not found on env_conf. Call env_conf.ensure_spaces() first.")
    return obs_space


def get_action_space(env_conf: Any) -> Any:
    """Get action space from env_conf.

    Args:
        env_conf: Environment configuration object

    Returns:
        Action space object with .shape attribute

    Raises:
        ValueError: If action space cannot be found
    """
    action_space = getattr(env_conf, "action_space", None)
    if action_space is None:
        raise ValueError("Action space not found on env_conf. Call env_conf.ensure_spaces() first.")
    return action_space


def get_obs_act_dims(env_conf: Any) -> tuple[int, int]:
    """Get observation and action dimensions from env_conf.

    Args:
        env_conf: Environment configuration object

    Returns:
        Tuple of (obs_dim, act_dim)

    Raises:
        ValueError: If spaces cannot be found
    """
    obs_space = get_obs_space(env_conf)
    action_space = get_action_space(env_conf)
    return int(obs_space.shape[0]), int(action_space.shape[0])
