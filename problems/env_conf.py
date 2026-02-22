import copy
import importlib
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

# Registry for Atari/DM-Control support. Only loaded when env_conf_atari_dm is imported.
_atari_dm_module = None


def register_atari_dm(module):
    """Register the Atari/DM-Control handler module. Called by env_conf_atari_dm on import."""
    global _atari_dm_module
    _atari_dm_module = module


def _get_atari_dm():
    if _atari_dm_module is None:
        raise RuntimeError("Atari/DM-Control support not loaded. Import problems.env_conf_atari_dm before using atari: or dm: tags.")
    return _atari_dm_module


def _gauss_policy_factory(variant: str, **kwargs):
    """Lazy factory for GaussianActorBackbonePolicy to avoid pulling rl into env_conf transitive deps."""

    def _factory(env_conf):
        from rl.policy_backbone import GaussianActorBackbonePolicyFactory

        return GaussianActorBackbonePolicyFactory(
            variant=variant,
            deterministic_eval=True,
            squash_mode="clip",
            init_log_std=-0.5,
            **kwargs,
        )(env_conf)

    return _factory


def _lazy_policy(module_name: str, class_name: str):
    def _factory(env_conf):
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls(env_conf)

    return _factory


def _normalize_dm_control_name(tag: str) -> str:
    if tag.startswith("dm:"):
        name = tag.split(":", 1)[1]
    else:
        name = tag.split("/", 1)[1]
    if not name.endswith("-v0") and not name.endswith("-v1"):
        name = f"{name}-v0"
    return f"dm_control/{name}"


def _parse_tag_options(tag, from_pixels, pixels_only):
    """Parse :fn, :pixels, :gauss, etc. from tag. Returns (tag, frozen_noise, policy_variant, from_pixels)."""
    frozen_noise = False
    policy_variant = None
    while ":" in tag:
        x = tag.split(":")
        opt = x[-1]
        if opt == "fn":
            frozen_noise = True
        elif opt == "pixels":
            from_pixels = True if from_pixels is None else from_pixels
        elif opt in ("gauss", "rl-gauss", "mlp16"):
            policy_variant = opt
        else:
            break
        tag = ":".join(x[:-1])
    return tag, frozen_noise, policy_variant, from_pixels


def _dm_control_policy_cls(use_pixels, policy_variant):
    adm = _get_atari_dm()
    if use_pixels:
        return adm.get_cnn_mlp_policy_factory()((32, 16))
    if policy_variant == "gauss":
        return _gauss_policy_factory(variant="rl-gauss-tanh")
    if policy_variant == "rl-gauss":
        return _gauss_policy_factory(variant="rl-gauss")
    return MLPPolicyFactory((32, 16))


def _atari_policy_cls(policy_variant):
    adm = _get_atari_dm()
    parsers = adm.get_atari_parsers_and_factories()
    _, AtariAgent57LiteFactory, AtariCNNPolicyFactory, AtariGaussianPolicyFactory = parsers
    if policy_variant == "agent57":
        return AtariAgent57LiteFactory(lstm_hidden=32, cnn_variant="small")
    if policy_variant == "gauss":
        return AtariGaussianPolicyFactory(
            hidden_sizes=(16, 16),
            cnn_latent_dim=64,
            variant="small",
            deterministic_eval=True,
            init_log_std=-0.5,
        )
    if policy_variant == "mlp16":
        return _lazy_policy("rl.policy_backbone", "AtariMLP16DiscretePolicy")
    return AtariCNNPolicyFactory((24,), variant="small")


def get_env_conf(
    tag,
    problem_seed=None,
    noise_level=None,
    noise_seed_0=None,
    from_pixels=None,
    pixels_only=None,
):
    tag, frozen_noise, policy_variant, from_pixels = _parse_tag_options(tag, from_pixels, pixels_only)
    pix_only = pixels_only if pixels_only is not None else True

    if tag.startswith("dm:") or tag.startswith("dm_control/"):
        env_name = _normalize_dm_control_name(tag)
        use_pixels = from_pixels if from_pixels is not None else False
        policy_cls = _dm_control_policy_cls(use_pixels, policy_variant)
        ec = EnvConf(
            env_name,
            gym_conf=GymConf(max_steps=1000, num_frames_skip=1, transform_state=False),
            policy_class=policy_cls,
            from_pixels=use_pixels,
            pixels_only=pix_only,
        )
    elif tag.startswith("atari:") or tag.startswith("ALE/"):
        _parse_atari_tag, _, _, _ = _get_atari_dm().get_atari_parsers_and_factories()
        env_id = _parse_atari_tag(tag) if "atari:" in tag or tag.startswith("ALE/") else tag
        policy_cls = _atari_policy_cls(policy_variant)
        ec = EnvConf(
            env_id,
            gym_conf=GymConf(max_steps=108000, num_frames_skip=1, transform_state=False),
            policy_class=policy_cls,
            from_pixels=True,
            pixels_only=True,
        )
    elif tag in _gym_env_confs:
        ec = copy.deepcopy(_gym_env_confs[tag])
    else:
        ec = EnvConf(
            tag,
            problem_seed=problem_seed,
            noise_level=noise_level,
            noise_seed_0=noise_seed_0,
            frozen_noise=frozen_noise,
        )

    ec.problem_seed = problem_seed
    ec.noise_seed_0 = noise_seed_0
    ec.frozen_noise = frozen_noise
    return ec


def default_policy(env_conf):
    if env_conf.gym_conf is not None:
        env_conf.ensure_spaces()
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

    # dm_control pixel observations (RL and BO)
    from_pixels: bool = False
    pixels_only: bool = True

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
        elif self.env_name.startswith("dm_control/"):
            make_dm_control_env = _get_atari_dm().get_dm_control_make()
            env = make_dm_control_env(
                self.env_name,
                from_pixels=getattr(self, "from_pixels", False),
                pixels_only=getattr(self, "pixels_only", True),
                **kwargs,
            )
        elif self.env_name.startswith("ALE/"):
            make_atari_env = _get_atari_dm().get_atari_make()
            render_mode = kwargs.get("render_mode")
            max_steps = int(self.gym_conf.max_steps) if self.gym_conf else 108000
            env = make_atari_env(
                self.env_name,
                render_mode=render_mode,
                max_episode_steps=max_steps,
            )
        elif self.gym_conf is not None:
            env = gym.make(self.env_name, **(kwargs | self.kwargs))
        else:
            env = other.make(self.env_name, problem_seed=self.problem_seed)

        return env

    def make(self, **kwargs):
        if self.gym_conf:
            self.ensure_spaces()
        env = self._make(**kwargs)
        if self.noise_level is not None:
            assert self.env_name[:2] in ["f:", "g:"], (
                "NYI: Noise is only supported for pure functions",
                self.env_name,
            )
            env = NoiseMaker(env, self.noise_level)
        return env

    def ensure_spaces(self):
        """Ensure state_space and action_space are populated (no-op if already set)."""
        if not self.gym_conf:
            return
        if self.gym_conf.state_space is not None and self.action_space is not None:
            return
        env = self._make()
        if self.gym_conf:
            self.gym_conf.state_space = env.observation_space
        self.action_space = env.action_space
        env.close()

    def __post_init__(self):
        if not self.kwargs:
            self.kwargs = {}
        if self.gym_conf:
            # Defer gym.make to avoid eagerly instantiating all envs at import time.
            self.gym_conf.state_space = None
            self.action_space = None
            return
        env = self._make()
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
    "cheetah": _gym_conf(
        "HalfCheetah-v5",
        policy_class=MLPPolicyFactory((32, 16)),
    ),
    "cheetah-16x16": _gym_conf(
        "HalfCheetah-v5",
        policy_class=MLPPolicyFactory((16, 16)),
    ),
    "cheetah-16x16-gauss": _gym_conf(
        "HalfCheetah-v5",
        policy_class=_gauss_policy_factory(variant="rl-gauss-tanh"),
    ),
    "cheetah-gauss": _gym_conf(
        "HalfCheetah-v5",
        policy_class=_gauss_policy_factory(variant="rl-gauss-small"),
    ),
    "reach": EnvConf("Reacher-v5", gym_conf=GymConf(max_steps=50)),
    # "push": EnvConf("Pusher-v4",  gym_conf=GymConf(max_steps=100)),
    "hop": _gym_conf("Hopper-v5"),
    "hop-gauss": _gym_conf(
        "Hopper-v5",
        policy_class=_gauss_policy_factory(variant="rl-gauss-small"),
    ),
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
        policy_class=_lazy_policy("problems.turbo_lunar_policy", "TurboLunarPolicy"),
    ),
}
