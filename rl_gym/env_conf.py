from dataclasses import dataclass
from typing import Any

import gymnasium as gym


def get_env_conf(tag, seed=None):
    ec = _env_confs[tag]
    ec.seed = seed
    ec.solved = 9999
    return ec


@dataclass
class EnvConf:
    env_name: str
    max_steps: int
    solved: int
    show_frames: int
    seed: int
    num_opt_0: int
    kwargs: dict = None
    k_state: float = 1.0
    state_space: Any = None
    action_space: Any = None

    def __post_init__(self):
        if not self.kwargs:
            self.kwargs = {}
        env = gym.make(self.env_name, **self.kwargs)
        self.state_space = env.observation_space
        self.action_space = env.action_space
        env.close()


_env_confs = {
    "mcc": EnvConf("MountainCarContinuous-v0", seed=None, max_steps=1000, solved=9999, show_frames=100, num_opt_0=100, k_state=10),
    "lunar": EnvConf("LunarLander-v2", seed=None, max_steps=500, kwargs={"continuous": True}, solved=999, show_frames=30, num_opt_0=100, k_state=10),
    "ant": EnvConf("Ant-v4", seed=None, max_steps=1000, solved=999, show_frames=30, num_opt_0=100, k_state=0.1),
    "bw": EnvConf("BipedalWalker-v3", seed=None, max_steps=1600, solved=300, show_frames=100, num_opt_0=100, k_state=0.1),
}
