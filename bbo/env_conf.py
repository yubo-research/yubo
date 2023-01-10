from dataclasses import dataclass
from typing import Any

import gym


@dataclass
class EnvConf:
    env_name: str
    max_steps: int
    solved: int
    show_frames: int
    seed: int
    kwargs: dict = None
    state_space: Any = None
    action_space: Any = None

    def __post_init__(self):
        if not self.kwargs:
            self.kwargs = {}
        env = gym.make(self.env_name, **self.kwargs)
        self.state_space = env.observation_space
        self.action_space = env.action_space
        env.close()
