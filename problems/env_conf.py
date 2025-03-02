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
    frozen_noise = False

    if ":" in tag:
        x = tag.split(":")
        opt = x[-1]
        # Ex: tlunar:fn
        if opt == "fn":
            frozen_noise = True
            tag = ":".join(x[:-1])
        else:
            assert len(x) == 2, (x, tag)

    if tag in _gym_env_confs:
        ec = copy.deepcopy(_gym_env_confs[tag])
        ec.problem_seed = problem_seed
        ec.noise_seed_0 = noise_seed_0
        ec.frozen_noise = frozen_noise
    else:
        ec = EnvConf(tag, problem_seed=problem_seed, noise_level=noise_level, noise_seed_0=noise_seed_0, frozen_noise=frozen_noise)

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
    bang_bang: bool = False


@dataclass
class EnvConf:
    env_name: str
    # Problem seed is changed once per repetition.
    # It is fixed for the duration of the optimization (all rounds).
    problem_seed: int
    policy_class: Any = None

    noise_level: float = None
    # The noise seed is changed once per run if num_denoise>0.
    # num_denoise=1 by default.
    noise_seed_0: int = None

    # If noise is frozen, then the same set of noise seeds
    #  is used in the denoising runs on every round.
    frozen_noise: bool = True

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


# See https://paperswithcode.com/task/openai-gym
_gym_env_confs = {
    # 95
    "mcc": EnvConf(
        "MountainCarContinuous-v0",
        problem_seed=None,
        gym_conf=GymConf(
            max_steps=1000,
            num_frames_skip=100,
            bang_bang=True,
        ),
    ),
    "pend": EnvConf("Pendulum-v1", problem_seed=None, gym_conf=GymConf(max_steps=200, num_frames_skip=100)),
    # 300
    "lunar": EnvConf(
        "LunarLander-v3",
        problem_seed=None,
        gym_conf=GymConf(
            max_steps=500,
            num_frames_skip=30,
            bang_bang=True,
        ),
        kwargs={"continuous": True},
    ),
    # 6600
    "ant": EnvConf("Ant-v5", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=30)),
    "mpend": EnvConf("InvertedPendulum-v5", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=30)),
    "macro": EnvConf("InvertedDoublePendulum-v5", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=30)),
    "swim": EnvConf("Swimmer-v5", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=30)),
    "reach": EnvConf("Reacher-v5", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=30)),
    # "push": EnvConf("Pusher-v4", problem_seed=None, gym_conf=GymConf(max_steps=100, num_frames_skip=30)),
    # 3300
    "hop": EnvConf("Hopper-v5", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=30)),
    # 6900
    "human": EnvConf("Humanoid-v5", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=30)),
    "stand": EnvConf("HumanoidStandup-v5", problem_seed=None, gym_conf=GymConf(max_steps=1000, num_frames_skip=30)),
    "bw": EnvConf(
        "BipedalWalker-v3",
        problem_seed=None,
        gym_conf=GymConf(
            max_steps=1600,
            num_frames_skip=100,
        ),
    ),
    "tlunar": EnvConf(
        # TuRBO paper specifies v2, but that raises an exception now
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
