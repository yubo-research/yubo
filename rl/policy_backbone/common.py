import torch.nn as nn

from rl.backbone import init_linear_layers


def init_linear(module: nn.Module) -> None:
    init_linear_layers(module, gain=0.5)


def obs_space_from_env_conf(env_conf):
    obs_space = getattr(env_conf, "state_space", None)
    if obs_space is None:
        gym_conf = getattr(env_conf, "gym_conf", None)
        obs_space = getattr(gym_conf, "state_space", None)
    if obs_space is None:
        raise ValueError("Observation space is missing on env_conf. Call env_conf.ensure_spaces() before creating policy_backbone policies.")
    return obs_space


def ensure_env_spaces(env_conf) -> None:
    if hasattr(env_conf, "ensure_spaces"):
        env_conf.ensure_spaces()


_obs_space_from_env_conf = obs_space_from_env_conf
_ensure_env_spaces = ensure_env_spaces
