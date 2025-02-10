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
from problems.new_problems import NNDraw, PestControl


def get_env_conf(tag, problem_seed=None, noise_level=None, noise_seed_0=None):
    if tag in _gym_env_confs:
        ec = copy.deepcopy(_gym_env_confs[tag])
        ec.problem_seed = problem_seed or 0
        ec.noise_seed_0 = noise_seed_0
    elif tag in _custom_env_confs:
        ec = copy.deepcopy(_custom_env_confs[tag])
        ec.problem_seed = problem_seed or 0
    else:
        ec = EnvConf(tag, problem_seed=problem_seed or 0, noise_level=noise_level, noise_seed_0=noise_seed_0)

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
            env = pure_functions.make(self.env_name, problem_seed=self.problem_seed)
        elif self.env_name[:2] == "g:":
            env = pure_functions.make(self.env_name, problem_seed=self.problem_seed, distort=False)
        elif self.gym_conf is not None:
            env = gym.make(self.env_name, **(kwargs | self.kwargs))
        elif self.env_name == "nndraw":
            env = NNDraw(dim=200, seed=self.problem_seed)
        elif self.env_name == "pest_control":
            env = PestControl(stages=25, categories=5, seed=self.problem_seed)
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
        if self.noise_seed_0 is None:
            self.noise_seed_0 = 0
        env = self._make()
        if self.gym_conf:
            self.gym_conf.state_space = getattr(env, "observation_space", None)
        self.action_space = getattr(env, "action_space", None)
        if hasattr(env, "close"):
            env.close()



_gym_env_confs = {
    "mcc": EnvConf("MountainCarContinuous-v0", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=100)),
    "pend": EnvConf("Pendulum-v1", problem_seed=None, gym_conf=GymConf(max_steps=200, num_frames_skip=100)),
    "lunar": EnvConf("LunarLander-v2", problem_seed=None, gym_conf=GymConf(max_steps=500, num_frames_skip=30), kwargs={"continuous": True}),
    "ant": EnvConf("Ant-v4", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=30)),
    "mpend": EnvConf("InvertedPendulum-v4", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=30)),
    "macro": EnvConf("InvertedDoublePendulum-v4", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=30)),
    "swim": EnvConf("Swimmer-v4", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=30)),
    "reach": EnvConf("Reacher-v4", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=30)),
    "push": EnvConf("Pusher-v4", problem_seed=None, gym_conf=GymConf(max_steps=100, num_frames_skip=30)),
    "hop": EnvConf("Hopper-v4", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=30)),
    "human": EnvConf("Humanoid-v4", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=30)),
    "stand": EnvConf("HumanoidStandup-v4", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=30)),
    "bw": EnvConf("BipedalWalker-v3", problem_seed=None, gym_conf=GymConf(max_steps=1600, num_frames_skip=100)),
    "tlunar": EnvConf(
        "LunarLander-v2",
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


_custom_env_confs = {
    "nndraw": EnvConf("nndraw", problem_seed=0),
    "pest_control": EnvConf("pest_control", problem_seed=0),
    #"mopta08": EnvConf("mopta08", problem_seed=0),
}