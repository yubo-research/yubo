import copy
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

import gymnasium as gym

import problems.other as other
import problems.pure_functions as pure_functions
from common.obs_mode import normalize_obs_mode, obs_mode_uses_pixels
from problems.bipedal_walker_policy import BipedalWalkerPolicy
from problems.gaussian_policy import GaussianPolicyFactory
from problems.linear_policy import LinearPolicy
from problems.mlp_policy import MLPPolicy, MLPPolicyFactory
from problems.mlp_torch_env import wrap_mlp_env
from problems.noise_maker import NoiseMaker
from problems.pure_function_policy import PureFunctionPolicy
from problems.turbo_lunar_policy import TurboLunarPolicy

_DM_CONTROL_DEFAULT_MAX_STEPS, _ATARI_DEFAULT_MAX_STEPS = 108000, 1000
_PURE_FUNCTION_MAX_STEPS = 1
_DEFAULT_MAX_STEPS = 99999


def _atari_env():
    ns: dict[str, Any] = {}
    exec("import problems.atari_env as m", ns)  # noqa: S102
    return ns["m"]


def _dm_control_env():
    ns: dict[str, Any] = {}
    exec("import problems.dm_control_env as m", ns)  # noqa: S102
    return ns["m"]


def _pixel_policies():
    ns: dict[str, Any] = {}
    exec("import problems.pixel_policies as m", ns)  # noqa: S102
    return ns["m"]


def _rl(*args, **kwargs):
    ns: dict[str, Any] = {}
    exec("from problems.rl_policy_factory import RLPolicyFactory as f", ns)  # noqa: S102
    return ns["f"](*args, **kwargs)


def _rl_gaussian(variant):
    ns: dict[str, Any] = {}
    exec("from problems.rl_policy_factory import gaussian_policy_factory as f", ns)  # noqa: S102
    return ns["f"](variant=variant)


def _parse_tag_options(tag, obs_mode):
    """Parse shared options from tag. Returns (tag, frozen_noise, obs_mode)."""
    frozen_noise = False
    while ":" in tag:
        tag, opt = tag.rsplit(":", 1)
        if opt == "fn":
            frozen_noise = True
        elif opt == "pixels":
            obs_mode = "image" if obs_mode is None else obs_mode
        else:
            tag = f"{tag}:{opt}"
            break
    return tag, frozen_noise, obs_mode


def _lookup_named_env_conf(tag):
    for env_confs in (_gym_env_confs, _dm_control_env_confs, _atari_env_confs):
        if tag in env_confs:
            return copy.deepcopy(env_confs[tag])
    return None


def _dm_dynamic_env_conf(tag, obs_mode):
    if not tag.startswith(("dm:", "dm_control/")):
        return None
    if tag.startswith(("dm:", "dm_control/")):
        mode = normalize_obs_mode(obs_mode)
        ec = _lookup_named_env_conf(tag)
        if ec is not None:
            ec.obs_mode = mode
            ec.max_steps = _DM_CONTROL_DEFAULT_MAX_STEPS if ec.max_steps is None else ec.max_steps
            return ec
        use_pixels = obs_mode_uses_pixels(mode)
        env_name = tag if tag.startswith("dm_control/") else f"dm_control/{tag.split(':', 1)[1]}"
        if not env_name.endswith(("-v0", "-v1")):
            env_name = f"{env_name}-v0"
        base, variant = tag.rsplit(":", 1) if tag.count(":") >= 2 and tag.rsplit(":", 1)[1] in {"gauss", "rl-gauss"} else (tag, None)
        if base != tag:
            env_name = (
                f"dm_control/{base.split(':', 1)[1]}-v0"
                if base.startswith("dm:") and not base.endswith(("-v0", "-v1"))
                else (base if base.startswith("dm_control/") else f"dm_control/{base.split(':', 1)[1]}")
            )
        if use_pixels:
            policy_class = _pixel_policies().CNNMLPPolicyFactory((32, 16))
        elif variant == "gauss":
            policy_class = _rl_gaussian("rl-gauss-tanh")
        elif variant == "rl-gauss":
            policy_class = _rl_gaussian("rl-gauss")
        else:
            policy_class = MLPPolicyFactory((32, 16))
        return EnvConf(
            env_name,
            policy_class=policy_class,
            obs_mode=mode,
            max_steps=_DM_CONTROL_DEFAULT_MAX_STEPS,
        )


def _atari_dynamic_env_conf(tag, obs_mode):
    if not tag.startswith(("atari:", "ALE/")):
        return None
    atari_env = _atari_env()
    pixel_policies = _pixel_policies()
    base, variant = tag.rsplit(":", 1) if tag.count(":") >= 2 and tag.rsplit(":", 1)[1] in {"agent57", "gauss", "mlp16"} else (tag, None)
    return EnvConf(
        atari_env.parse_tag(base),
        policy_class=atari_env.policy_class(
            policy_variant=variant,
            atari_agent57_factory=pixel_policies.AtariAgent57LiteFactory,
            atari_cnn_policy_factory=pixel_policies.AtariCNNPolicyFactory,
            atari_gaussian_policy_factory=pixel_policies.AtariGaussianPolicyFactory,
        ),
        obs_mode=normalize_obs_mode(obs_mode or "image"),
        max_steps=_ATARI_DEFAULT_MAX_STEPS,
    )


def _resolve_backend_env_conf(tag, obs_mode):
    ec = _dm_dynamic_env_conf(tag, obs_mode)
    if ec is not None:
        return ec
    return _atari_dynamic_env_conf(tag, obs_mode)


def get_env_conf(
    tag,
    problem_seed=None,
    noise_level=None,
    noise_seed_0=None,
    obs_mode=None,
    atari_preprocess=None,
):
    tag, frozen_noise, obs_mode = _parse_tag_options(tag, obs_mode)
    tag = "atari:Pong" if tag == "atari-pong" else tag

    ec = _lookup_named_env_conf(tag)
    if ec is None and (ec := _resolve_backend_env_conf(tag, obs_mode)) is None:
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
    ec.obs_mode = normalize_obs_mode(obs_mode if obs_mode is not None else getattr(ec, "obs_mode", "vector"))
    if atari_preprocess is not None:
        if not isinstance(atari_preprocess, dict):
            raise TypeError("atari_preprocess must be a dict when provided.")
        if not str(ec.env_name).startswith("ALE/"):
            raise ValueError("atari_preprocess is only valid for Atari envs (ALE/*).")
        ec.atari_preprocess = copy.deepcopy(atari_preprocess)
    return ec


def default_policy(env_conf):
    has_policy_or_gym = env_conf.policy_class is not None or env_conf.gym_conf is not None
    missing_spaces = getattr(env_conf, "state_space", None) is None or getattr(env_conf, "action_space", None) is None
    if has_policy_or_gym and missing_spaces:
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

    obs_mode: str = "vector"
    # Optional ALE preprocessing/runtime overrides (applies only to ALE envs).
    atari_preprocess: dict[str, Any] | None = None

    noise_level: float = None
    # The noise seed is changed once per run if num_denoise>0.
    # num_denoise=1 by default.
    noise_seed_0: int = None

    # If noise is frozen, then the same set of noise seeds
    #  is used in the denoising runs on every round.
    frozen_noise: bool = True

    gym_conf: GymConf = None
    state_space: Any = None
    max_steps: int | None = None
    action_space: Any = None
    kwargs: dict = None

    def _make(self, **kwargs):
        if self.env_name[:2] == "f:":
            env = pure_functions.make(self.env_name, problem_seed=self.problem_seed, distort=True)
        elif self.env_name[:2] == "g:":
            env = pure_functions.make(self.env_name, problem_seed=self.problem_seed, distort=False)
        elif self.env_name.startswith("dm_control/"):
            env = _dm_control_env().make(
                self.env_name,
                obs_mode=getattr(self, "obs_mode", "state"),
                **kwargs,
            )
        elif self.env_name.startswith("ALE/"):
            atari_env = _atari_env()
            if self.max_steps is None:
                raise ValueError("EnvConf.max_steps must be set for ALE environments.")
            preprocess = kwargs.pop("preprocess", None)
            if preprocess is None:
                default_preprocess = atari_env.AtariPreprocessOptions()
                preprocess_kwargs = asdict(default_preprocess) if is_dataclass(default_preprocess) else dict(vars(default_preprocess))
                if isinstance(self.atari_preprocess, dict):
                    preprocess_kwargs.update(self.atari_preprocess)
                preprocess = atari_env.AtariPreprocessOptions(**preprocess_kwargs)
            env = atari_env.make_atari_env(
                self.env_name,
                render_mode=kwargs.get("render_mode"),
                max_episode_steps=int(self.max_steps),
                preprocess=preprocess,
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
        if self.state_space is not None and self.action_space is not None:
            return
        if self.gym_conf is not None and self.gym_conf.state_space is not None and self.action_space is not None:
            self.state_space = self.gym_conf.state_space
            return
        env = self._make()
        self.state_space = env.observation_space
        if self.gym_conf is not None:
            self.gym_conf.state_space = self.state_space
        self.action_space = env.action_space
        env.close()

    def __post_init__(self):
        self.obs_mode = normalize_obs_mode(self.obs_mode)
        if not self.kwargs:
            self.kwargs = {}
        if self.env_name[:2] in ("f:", "g:") and self.max_steps is None:
            self.max_steps = _PURE_FUNCTION_MAX_STEPS
        if self.max_steps is None and self.gym_conf is None and not self.env_name.startswith(("ALE/", "dm_control/")):
            self.max_steps = _DEFAULT_MAX_STEPS
        if self.gym_conf:
            # Defer gym.make to avoid eagerly instantiating all envs at import time.
            self.gym_conf.state_space = None
            self.state_space = None
            self.action_space = None
            return
        if self.env_name.startswith(("ALE/", "dm_control/")):
            # Defer heavy env creation for Atari/DM control until first use.
            self.state_space = None
            self.action_space = None
            return
        env = self._make()
        self.state_space = env.observation_space
        self.action_space = env.action_space
        env.close()

    def make_torch_env(self, **kwargs):
        """Create environment with torch module exposed for direct perturbation.

        For environments with MLP policies, this creates a wrapped environment
        that exposes the policy module via torch_env().module for use with
        BSZO and other UHD optimizers requiring direct parameter access.
        """
        if self.state_space is None or self.action_space is None:
            self.ensure_spaces()

        # Check if this is an MLP policy that can be directly perturbed
        if self.policy_class is not None:
            # Create the policy and check if it's an MLPPolicy
            policy = self.policy_class(self)
            if isinstance(policy, MLPPolicy):
                if self.gym_conf is None:
                    raise ValueError("make_torch_env for MLPPolicy requires a gym_conf with max_steps and num_frames_skip.")
                # Create the base gym environment
                env = self._make(**kwargs)
                # Wrap with torch env wrapper
                return wrap_mlp_env(
                    env=env,
                    policy=policy,
                    max_steps=self.gym_conf.max_steps if self.gym_conf else 1000,
                    num_frames_skip=self.gym_conf.num_frames_skip if self.gym_conf else 1,
                )
            # If not MLPPolicy, close any resources and fall through
            if hasattr(policy, "close"):
                policy.close()

        # For non-MLP policies, fall back to standard make
        return self.make(**kwargs)


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
    "cheetah-small-1k2": _gym_conf(
        "HalfCheetah-v5",
        policy_class=MLPPolicyFactory((32, 16), use_layer_norm=False, activation="tanh"),
    ),
    "cheetah-cleanrl": _gym_conf(
        "HalfCheetah-v5",
        policy_class=_rl(
            MLPPolicyFactory((64, 64), use_layer_norm=False, activation="tanh"),
            critic=MLPPolicyFactory((64, 64), use_layer_norm=False, activation="tanh"),
            share_backbone=False,
        ),
    ),
    # CleanRL-style (separate actor/critic) with 16×16 hidden
    "cheetah-cleanrl-16x16": _gym_conf(
        "HalfCheetah-v5",
        policy_class=_rl(
            MLPPolicyFactory((16, 16), use_layer_norm=False, activation="tanh"),
            critic=MLPPolicyFactory((16, 16), use_layer_norm=False, activation="tanh"),
            share_backbone=False,
        ),
    ),
    # Same as cheetah-cleanrl-16x16 but with layer norm
    "cheetah-cleanrl-16x16-ln": _gym_conf(
        "HalfCheetah-v5",
        policy_class=_rl(
            MLPPolicyFactory((16, 16), use_layer_norm=True, activation="tanh"),
            critic=MLPPolicyFactory((16, 16), use_layer_norm=True, activation="tanh"),
            share_backbone=False,
        ),
    ),
    # CleanRL ppo_continuous_action: HalfCheetah-v4, 64x64 tanh (exact match)
    "cheetah-v4-cleanrl": _gym_conf(
        "HalfCheetah-v4",
        policy_class=_rl(
            MLPPolicyFactory((64, 64), use_layer_norm=False, activation="tanh"),
            critic=MLPPolicyFactory((64, 64), use_layer_norm=False, activation="tanh"),
            share_backbone=False,
        ),
    ),
    "cheetah-v4": _gym_conf(
        "HalfCheetah-v4",
        policy_class=MLPPolicyFactory((32, 16)),
    ),
    # GAC paper Appendix C.1: 256x256 ReLU backbone
    "cheetah-v4-gac": _gym_conf(
        "HalfCheetah-v4",
        policy_class=_rl(
            MLPPolicyFactory((256, 256), use_layer_norm=False, activation="relu"),
            critic=MLPPolicyFactory((256, 256), use_layer_norm=False, activation="relu"),
            share_backbone=False,
        ),
    ),
    "cheetah-v5-gac": _gym_conf(
        "HalfCheetah-v5",
        policy_class=_rl(
            MLPPolicyFactory((256, 256), use_layer_norm=False, activation="relu"),
            critic=MLPPolicyFactory((256, 256), use_layer_norm=False, activation="relu"),
            share_backbone=False,
        ),
    ),
    "quadruped-run-64x64": _gym_conf(
        "dm_control/quadruped-run-v0",
        policy_class=MLPPolicyFactory((64, 64)),
    ),
    "cheetah-16x16": _gym_conf(
        "HalfCheetah-v5",
        policy_class=MLPPolicyFactory((16, 16)),
    ),
    # BO-only: 16x32 LN actor, ~1,064 params
    "cheetah-16x32-ln": _gym_conf(
        "HalfCheetah-v5",
        policy_class=MLPPolicyFactory((16, 32), use_layer_norm=True, activation="tanh"),
    ),
    "cheetah-16x16-gauss": _gym_conf(
        "HalfCheetah-v5",
        policy_class=GaussianPolicyFactory(
            variant="rl-gauss-tanh",
            deterministic_eval=True,
            squash_mode="clip",
            init_log_std=-0.5,
        ),
    ),
    "cheetah-gauss": _gym_conf(
        "HalfCheetah-v5",
        policy_class=GaussianPolicyFactory(
            variant="rl-gauss-small",
            deterministic_eval=True,
            squash_mode="clip",
            init_log_std=-0.5,
        ),
    ),
    "reach": EnvConf("Reacher-v5", gym_conf=GymConf(max_steps=50)),
    # "push": EnvConf("Pusher-v4",  gym_conf=GymConf(max_steps=100)),
    "hop": _gym_conf("Hopper-v5"),
    "hop-gauss": _gym_conf(
        "Hopper-v5",
        policy_class=GaussianPolicyFactory(
            variant="rl-gauss-small",
            deterministic_eval=True,
            squash_mode="clip",
            init_log_std=-0.5,
        ),
    ),
    # 6900
    "human": _gym_conf("Humanoid-v5"),
    # 130,000 - https://arxiv.org/html/2304.12778
    "stand": _gym_conf("HumanoidStandup-v5"),
    "stand-mlp": _gym_conf(
        "HumanoidStandup-v5",
        policy_class=MLPPolicyFactory((32, 16)),
    ),
    "stand-mlp2": _gym_conf(
        "HumanoidStandup-v5",
        policy_class=MLPPolicyFactory((256, 128)),
    ),
    "stand-mlp3": _gym_conf(
        "HumanoidStandup-v5",
        policy_class=MLPPolicyFactory((1024, 600)),
    ),
    "stand-mlp4": _gym_conf(
        "HumanoidStandup-v5",
        policy_class=MLPPolicyFactory((4096, 2060)),
    ),
    "stand-mlp5": _gym_conf(
        "HumanoidStandup-v5",
        policy_class=MLPPolicyFactory((32000, 31000)),
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
        policy_class=MLPPolicyFactory((1024, 512, 256, 128)),
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


_dm_control_env_confs = {
    "dm_control/quadruped-run-v0-10k": EnvConf(
        "dm_control/quadruped-run-v0",
        policy_class=_rl(
            MLPPolicyFactory((64, 64), use_layer_norm=True, activation="silu"),
            critic=MLPPolicyFactory((128, 128), use_layer_norm=True, activation="silu"),
        ),
        max_steps=_DM_CONTROL_DEFAULT_MAX_STEPS,
    ),
    "dm_control/quadruped-run-v0-small": EnvConf(
        "dm_control/quadruped-run-v0",
        policy_class=MLPPolicyFactory((4, 4)),
        max_steps=_DM_CONTROL_DEFAULT_MAX_STEPS,
    ),
    "dm_control/quadruped-run-v0-100k": EnvConf(
        "dm_control/quadruped-run-v0",
        policy_class=MLPPolicyFactory((256, 256), use_layer_norm=True, activation="silu"),
        max_steps=_DM_CONTROL_DEFAULT_MAX_STEPS,
    ),
    "dm_control/quadruped-run-v0-1m": EnvConf(
        "dm_control/quadruped-run-v0",
        policy_class=MLPPolicyFactory((1024, 1024), use_layer_norm=True, activation="silu"),
        max_steps=_DM_CONTROL_DEFAULT_MAX_STEPS,
    ),
}


_atari_env_confs: dict[str, EnvConf] = {}
