from dataclasses import dataclass
from typing import Any

import gymnasium as gym

import problems.pure_functions as pure_functions
from problems.linear_policy import LinearPolicy
from problems.noise_maker import NoiseMaker
from problems.pure_function_policy import PureFunctionPolicy
from problems.turbo_lunar_policy import TurboLunarPolicy


def get_env_conf(tag, problem_seed=None, noise_level=None, noise_seed_0=None):
    if tag in _env_confs:
        ec = _env_confs[tag]
        ec.problem_seed = problem_seed
        ec.noise_seed_0 = noise_seed_0
        ec.solved = 9999
    else:
        ec = EnvConf(tag, problem_seed=problem_seed, noise_level=noise_level, noise_seed_0=noise_seed_0, max_steps=1000, solved=9999)

    return ec


def default_policy(env_conf):
    if env_conf.policy_class is not None:
        return env_conf.policy_class(env_conf)
    elif env_conf.env_name[:2] == "f:":
        return PureFunctionPolicy(env_conf)
    else:
        return LinearPolicy(env_conf)


@dataclass
class EnvConf:
    env_name: str
    max_steps: int
    solved: int
    problem_seed: int
    noise_level: float = None
    noise_seed_0: int = None
    show_frames: int = None
    kwargs: dict = None
    state_space: Any = None
    action_space: Any = None
    policy_class: Any = None
    transform: bool = True

    def _make(self, **kwargs):
        if self.env_name[:2] == "f:":
            env = pure_functions.make(self.env_name, problem_seed=self.problem_seed)
        else:
            env = gym.make(self.env_name, **(kwargs | self.kwargs))
        return env

    def make(self, **kwargs):
        env = self._make(**kwargs)
        if self.noise_level is not None:
            assert self.env_name[:2] == "f:", ("NYI: Noise is only supported for pure functions", self.env_name)
            env = NoiseMaker(env, self.noise_level)
        return env

    def __post_init__(self):
        if not self.kwargs:
            self.kwargs = {}
        env = self._make()
        self.state_space = env.observation_space
        self.action_space = env.action_space
        env.close()


_env_confs = {
    "mcc": EnvConf("MountainCarContinuous-v0", problem_seed=None, max_steps=1000, solved=9999, show_frames=100),
    "pend": EnvConf("Pendulum-v1", problem_seed=None, max_steps=200, solved=9999, show_frames=100),
    "lunar": EnvConf("LunarLander-v2", problem_seed=None, max_steps=500, kwargs={"continuous": True}, solved=999, show_frames=30),
    "ant": EnvConf("Ant-v4", problem_seed=None, max_steps=1000, solved=999, show_frames=30),
    "mpend": EnvConf("InvertedPendulum-v4", problem_seed=None, max_steps=1000, solved=999, show_frames=30),
    "macro": EnvConf("InvertedDoublePendulum-v4", problem_seed=None, max_steps=1000, solved=999, show_frames=30),
    "swim": EnvConf("Swimmer-v4", problem_seed=None, max_steps=1000, solved=999, show_frames=30),
    "reach": EnvConf("Reacher-v4", problem_seed=None, max_steps=1000, solved=999, show_frames=30),
    "push": EnvConf("Pusher-v4", problem_seed=None, max_steps=100, solved=999, show_frames=30),
    "hop": EnvConf("Hopper-v4", problem_seed=None, max_steps=1000, solved=999, show_frames=30),
    "human": EnvConf("Humanoid-v4", problem_seed=None, max_steps=1000, solved=999, show_frames=30),
    "stand": EnvConf("HumanoidStandup-v4", problem_seed=None, max_steps=1000, solved=999, show_frames=30),
    "bw": EnvConf("BipedalWalker-v3", problem_seed=None, max_steps=1600, solved=300, show_frames=100),
    "tlunar": EnvConf(
        "LunarLander-v2",
        problem_seed=None,
        max_steps=500,
        kwargs={"continuous": False},
        solved=999,
        show_frames=30,
        policy_class=TurboLunarPolicy,
        transform=False,
    ),
}
