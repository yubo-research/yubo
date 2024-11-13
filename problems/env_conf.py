import copy
from dataclasses import dataclass
from typing import Any

import gymnasium as gym

import problems.other as other
import problems.pure_functions as pure_functions
from problems.linear_policy import LinearPolicy
from problems.noise_maker import NoiseMaker
from problems.pure_function_policy import PureFunctionPolicy
from problems.turbo_lunar_policy import TurboLunarPolicy


def get_env_conf(tag, problem_seed=None, noise_level=None, noise_seed_0=None):
    if tag in _gym_env_confs:
        ec = copy.deepcopy(_gym_env_confs[tag])
        ec.problem_seed = problem_seed
        ec.noise_seed_0 = noise_seed_0
    else:
        ec = EnvConf(tag, problem_seed=problem_seed, noise_level=noise_level, noise_seed_0=noise_seed_0)

    return ec


def default_policy(env_conf):
    if env_conf.policy_class is not None:
        return env_conf.policy_class(env_conf)
    elif env_conf.gym_conf is not None:
        return LinearPolicy(env_conf)
    else:  # env_conf.env_name[:2] == "f:":
        return PureFunctionPolicy(env_conf)


@dataclass
class GymConf:
    max_steps: int = None
    num_frames_skip: int = None
    state_space: Any = None
    transform_state: bool = True


@dataclass
class EnvConf:
    env_name: str
    problem_seed: int
    policy_class: Any = None

    noise_level: float = None
    noise_seed_0: int = None

    gym_conf: GymConf = None
    action_space: Any = None
    kwargs: dict = None

    def _make(self, **kwargs):
        if self.env_name[:2] == "f:":
            env = pure_functions.make(self.env_name, problem_seed=self.problem_seed, distort=True)
        elif self.env_name[:2] == "g:":
            env = pure_functions.make(self.env_name, problem_seed=self.problem_seed, distort=False)
        elif self.gym_conf is not None:
            env = gym.make(self.env_name, **(kwargs | self.kwargs))
        else:
            env = other.make(self.env_name, problem_seed=self.problem_seed)

        return env

    def make(self, **kwargs):
        env = self._make(**kwargs)
        if self.noise_level is not None:
            assert self.env_name[:2] in ["f:", "g:"], ("NYI: Noise is only supported for pure functions", self.env_name)
            env = NoiseMaker(env, self.noise_level)
        return env

    def __post_init__(self):
        if not self.kwargs:
            self.kwargs = {}
        env = self._make()
        if self.gym_conf:
            self.gym_conf.state_space = env.observation_space
        self.action_space = env.action_space
        env.close()


_gym_env_confs = {
    "mcc": EnvConf("MountainCarContinuous-v0", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=100)),
    "pend": EnvConf("Pendulum-v1", problem_seed=None, gym_conf=GymConf(max_steps=200, num_frames_skip=100)),
    "lunar": EnvConf("LunarLander-v3", problem_seed=None, gym_conf=GymConf(max_steps=500, num_frames_skip=30), kwargs={"continuous": True}),
    "ant": EnvConf("Ant-v5", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=30)),
    "mpend": EnvConf("InvertedPendulum-v5", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=30)),
    "macro": EnvConf("InvertedDoublePendulum-v5", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=30)),
    "swim": EnvConf("Swimmer-v5", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=30)),
    "reach": EnvConf("Reacher-v5", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=30)),
    # "push": EnvConf("Pusher-v4", problem_seed=None, gym_conf=GymConf(max_steps=100, num_frames_skip=30)),
    "hop": EnvConf("Hopper-v5", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=30)),
    "human": EnvConf("Humanoid-v5", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=30)),
    "stand": EnvConf("HumanoidStandup-v5", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=30)),
    "bw": EnvConf("BipedalWalker-v3", problem_seed=None, gym_conf=GymConf(max_steps=1600, num_frames_skip=100)),
    "tlunar": EnvConf(
        "LunarLander-v3",
        problem_seed=None,
        gym_conf=GymConf(
            max_steps=500,
            num_frames_skip=30,
            transform_state=False,
        ),
        kwargs={"continuous": False},
        policy_class=TurboLunarPolicy,
    ),
}
