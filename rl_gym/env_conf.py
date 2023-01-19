from dataclasses import dataclass
from typing import Any

import gymnasium as gym

import rl_gym.pure_functions as pure_functions
from rl_gym.linear_policy import LinearPolicy
from rl_gym.pure_function_policy import PureFunctionPolicy


def get_env_conf(tag, seed=None):
    if tag[:2] == "f:":
        ec = EnvConf(tag, seed=None, max_steps=1000, solved=-0.01, num_opt_0=100)
    else:
        ec = _env_confs[tag]
    ec.seed = seed
    ec.solved = 9999
    return ec


def default_policy(env_conf):
    if env_conf.env_name[:2] == "f:":
        return PureFunctionPolicy(env_conf)
    else:
        return LinearPolicy(env_conf)


@dataclass
class EnvConf:
    env_name: str
    max_steps: int
    solved: int
    seed: int
    show_frames: int = None
    num_opt_0: int = None
    kwargs: dict = None
    k_state: float = 1.0
    state_space: Any = None
    action_space: Any = None

    def make(self, **kwargs):
        if self.env_name[:2] == "f:":
            return pure_functions.make(self.env_name)
        return gym.make(self.env_name, **(kwargs | self.kwargs))

    def __post_init__(self):
        if not self.kwargs:
            self.kwargs = {}
        env = self.make()
        self.state_space = env.observation_space
        self.action_space = env.action_space
        env.close()


_env_confs = {
    "mcc": EnvConf("MountainCarContinuous-v0", seed=None, max_steps=1000, solved=9999, show_frames=100, num_opt_0=100, k_state=10),
    "lunar": EnvConf("LunarLander-v2", seed=None, max_steps=500, kwargs={"continuous": True}, solved=999, show_frames=30, num_opt_0=100, k_state=10),
    "ant": EnvConf("Ant-v4", seed=None, max_steps=1000, solved=999, show_frames=30, num_opt_0=100, k_state=0.1),
    "bw": EnvConf("BipedalWalker-v3", seed=None, max_steps=1600, solved=300, show_frames=100, num_opt_0=100, k_state=0.1),
}
