from dataclasses import dataclass
from typing import Any

import gymnasium as gym

import problems.pure_functions as pure_functions
from problems.linear_policy import LinearPolicy
from problems.noise_maker import NoiseMaker
from problems.pure_function_policy import PureFunctionPolicy


def get_env_conf(tag, seed=None, noise=None):
    if tag in _env_confs:
        ec = _env_confs[tag]
        ec.seed = seed
        ec.solved = 9999
    else:
        ec = EnvConf(tag, seed=seed, noise=noise, max_steps=1000, solved=9999)

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
    noise: float = None
    show_frames: int = None
    kwargs: dict = None
    state_space: Any = None
    action_space: Any = None

    def _make(self, new_seed=None, **kwargs):
        if self.env_name[:2] == "f:":
            if new_seed is None:
                seed = self.seed
            else:
                seed = new_seed
            env = pure_functions.make(self.env_name, seed=seed)
        else:
            env = gym.make(self.env_name, **(kwargs | self.kwargs))
        return env

    def make(self, **kwargs):
        env = self._make(**kwargs)
        if self.noise is not None:
            assert self.env_name[:2] == "f:", ("NYI: Noise is only supported for pure functions", self.env_name)
            env = NoiseMaker(env, self.noise)
        return env

    def __post_init__(self):
        if not self.kwargs:
            self.kwargs = {}
        env = self._make()
        self.state_space = env.observation_space
        self.action_space = env.action_space
        env.close()


_env_confs = {
    "mcc": EnvConf("MountainCarContinuous-v0", seed=None, max_steps=1000, solved=9999, show_frames=100),
    "lunar": EnvConf("LunarLander-v2", seed=None, max_steps=500, kwargs={"continuous": True}, solved=999, show_frames=30),
    "ant": EnvConf("Ant-v4", seed=None, max_steps=1000, solved=999, show_frames=30),
    "bw": EnvConf("BipedalWalker-v3", seed=None, max_steps=1600, solved=300, show_frames=100),
}
