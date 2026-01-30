import copy
from dataclasses import dataclass
from typing import Any

import gymnasium as gym

import problems.other as other
import problems.pure_functions as pure_functions
from problems.bipedal_walker_policy import BipedalWalkerPolicy
from problems.linear_policy import LinearPolicy
from problems.mlp_policy import MLPPolicyFactory
from problems.noise_maker import NoiseMaker
from problems.pure_function_policy import PureFunctionPolicy
from problems.turbo_lunar_policy import TurboLunarPolicy


def get_env_conf(tag, problem_seed=None, noise_level=None, noise_seed_0=None):
    frozen_noise = False

    if ":" in tag:
        x = tag.split(":")
        opt = x[-1]
        if opt == "fn":
            frozen_noise = True
            tag = ":".join(x[:-1])

    if tag in _gym_env_confs:
        ec = copy.deepcopy(_gym_env_confs[tag])
        ec.problem_seed = problem_seed
        ec.noise_seed_0 = noise_seed_0
        ec.frozen_noise = frozen_noise
    else:
        ec = EnvConf(
            tag,
            problem_seed=problem_seed,
            noise_level=noise_level,
            noise_seed_0=noise_seed_0,
            frozen_noise=frozen_noise,
        )

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
    max_steps: int = 1000
    num_frames_skip: int = 30
    state_space: Any = None
    transform_state: bool = True


@dataclass
class EnvConf:
    env_name: str
    # Problem seed is changed once per repetition.
    # It is fixed for the duration of the optimization (all rounds).
    problem_seed: int = None
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
            assert self.env_name[:2] in ["f:", "g:"], (
                "NYI: Noise is only supported for pure functions",
                self.env_name,
            )
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


def _gym_conf(env_name, gym_conf=None, policy_class=None, kwargs=None, noise_seed_0=None):
    if gym_conf is None:
        gym_conf = GymConf()

    return EnvConf(
        env_name,
        gym_conf=gym_conf,
        policy_class=policy_class,
        kwargs=kwargs,
        noise_seed_0=noise_seed_0,
    )


# See https://paperswithcode.com/task/openai-gym
# num_frames_skip is not "frame_skip" in gymnasium. num_frames_skip is only used internally.
_gym_env_confs = {
    # 95
    "mcc": _gym_conf(
        "MountainCarContinuous-v0",
        gym_conf=GymConf(num_frames_skip=100),
    ),
    "pend": EnvConf("Pendulum-v1", gym_conf=GymConf(max_steps=200, num_frames_skip=100)),
    # 3580 - https://arxiv.org/pdf/1803.07055
    # 6600 - 2024 [??ref] k
    "ant": _gym_conf("Ant-v5"),
    "mpend": _gym_conf("InvertedPendulum-v5"),
    "macro": _gym_conf("InvertedDoublePendulum-v5"),
    # 325 - https://arxiv.org/pdf/1803.07055
    "swim": _gym_conf("Swimmer-v5"),
    "reach": EnvConf("Reacher-v5", gym_conf=GymConf(max_steps=50)),
    # "push": EnvConf("Pusher-v4",  gym_conf=GymConf(max_steps=100)),
    "hop": _gym_conf("Hopper-v5"),
    # 6900
    "human": _gym_conf("Humanoid-v5"),
    # 130,000 - https://arxiv.org/html/2304.12778
    "stand": _gym_conf("HumanoidStandup-v5"),
    "stand-mlp": _gym_conf(
        "HumanoidStandup-v5",
        policy_class=MLPPolicyFactory((32, 16)),
    ),
    "bw": _gym_conf(
        "BipedalWalker-v3",
        gym_conf=GymConf(
            max_steps=1600,
            num_frames_skip=100,
        ),
    ),
    "bw-linraw": _gym_conf(
        "BipedalWalker-v3",
        gym_conf=GymConf(
            max_steps=1600,
            num_frames_skip=100,
            transform_state=False,
        ),
    ),
    # See https://github.com/hardmaru/estool/blob/b0954523e906d852287c6f515f34756c550ccf42/config.py#L309
    #  for config (i.e., (40,40))
    # https://arxiv.org/html/2304.12778 uses (16,)
    #
    "bw-mlp": _gym_conf(
        "BipedalWalker-v3",
        gym_conf=GymConf(
            max_steps=1600,
            num_frames_skip=100,
        ),
        policy_class=MLPPolicyFactory((), rnn_hidden_size=4, use_layer_norm=True, use_prev_action=True),
    ),
    "bw-heur": _gym_conf(
        "BipedalWalker-v3",
        gym_conf=GymConf(
            max_steps=1600,
            num_frames_skip=100,
            transform_state=False,
        ),
        policy_class=BipedalWalkerPolicy,
        noise_seed_0=1,
    ),
    # 300
    "lunar": _gym_conf(
        "LunarLander-v3",
        gym_conf=GymConf(
            max_steps=500,
        ),
        kwargs={"continuous": True},
    ),
    # 300
    "lunar-mlp": _gym_conf(
        "LunarLander-v3",
        gym_conf=GymConf(
            max_steps=500,
        ),
        kwargs={"continuous": True},
        policy_class=MLPPolicyFactory((16, 8)),
    ),
    "tlunar": EnvConf(
        # TuRBO paper specifies v2, but that raises an exception now
        "LunarLander-v3",
        gym_conf=GymConf(
            max_steps=500,
            transform_state=False,
        ),
        kwargs={"continuous": False},
        policy_class=TurboLunarPolicy,
    ),
}
