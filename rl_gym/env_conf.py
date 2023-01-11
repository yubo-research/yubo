from dataclasses import dataclass
from typing import Any

import gymnasium as gym


@dataclass
class EnvConf:
    env_name: str
    max_steps: int
    solved: int
    show_frames: int
    seed: int
    num_opt_0: int
    kwargs: dict = None
    k_action: float = 1.0
    state_space: Any = None
    action_space: Any = None

    def __post_init__(self):
        if not self.kwargs:
            self.kwargs = {}
        env = gym.make(self.env_name, **self.kwargs)
        self.state_space = env.observation_space
        self.action_space = env.action_space
        env.close()
