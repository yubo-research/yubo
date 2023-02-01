from dataclasses import dataclass
from typing import Any

import gymnasium as gym

import problems.pure_functions as pure_functions
from problems.linear_policy import LinearPolicy
from problems.pure_function_policy import PureFunctionPolicy


def get_env_conf(tag, seed=None):
    if tag[:2] == "f:":
        ec = EnvConf(tag, seed=None, max_steps=1000, solved=-0.01)
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
    kwargs: dict = None
    state_space: Any = None
    action_space: Any = None

    def make(self, **kwargs):
        if self.env_name[:2] == "f:":
            return pure_functions.make(self.env_name, seed=self.seed)
        return gym.make(self.env_name, **(kwargs | self.kwargs))

    def __post_init__(self):
        if not self.kwargs:
            self.kwargs = {}
        env = self.make()
        self.state_space = env.observation_space
        self.action_space = env.action_space
        env.close()


_env_confs = {
    "mcc": EnvConf("MountainCarContinuous-v0", seed=None, max_steps=1000, solved=9999, show_frames=100),
    "lunar": EnvConf("LunarLander-v2", seed=None, max_steps=500, kwargs={"continuous": True}, solved=999, show_frames=30),
    "ant": EnvConf("Ant-v4", seed=None, max_steps=1000, solved=999, show_frames=30),
    "bw": EnvConf("BipedalWalker-v3", seed=None, max_steps=1600, solved=300, show_frames=100),
}
