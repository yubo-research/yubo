import copy
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable

import gymnasium as gym

import problems.other as other
import problems.pure_functions as pure_functions
from problems.bipedal_walker_policy import BipedalWalkerPolicy
from problems.linear_policy import LinearPolicy
from problems.mlp_policy import MLPPolicy, MLPPolicyFactory
from problems.mlp_torch_env import wrap_mlp_env
from problems.noise_maker import NoiseMaker
from problems.pure_function_policy import PureFunctionPolicy
from problems.turbo_lunar_policy import TurboLunarPolicy

_DM_CONTROL_DEFAULT_MAX_STEPS = 1000
_ATARI_DEFAULT_MAX_STEPS = 108000
_PURE_FUNCTION_MAX_STEPS = 1
_DEFAULT_MAX_STEPS = 99999

_ATARI_DM_BINDINGS = None
_ATARI_DM_BINDINGS_LOADER: Callable[[], Any] | None = None


def register_atari_dm_bindings_loader(loader: Callable[[], Any]) -> None:
    global _ATARI_DM_BINDINGS_LOADER
    _ATARI_DM_BINDINGS_LOADER = loader


def _get_atari_dm_bindings():
    global _ATARI_DM_BINDINGS
    if _ATARI_DM_BINDINGS is None:
        if _ATARI_DM_BINDINGS_LOADER is None:
            raise RuntimeError("Atari/DM bindings are not registered. Call problems.env_conf_backends.register_with_env_conf() before using Atari/DM env tags.")
        _ATARI_DM_BINDINGS = _ATARI_DM_BINDINGS_LOADER()
    return _ATARI_DM_BINDINGS


def _parse_tag_options(tag, from_pixels):
    """Parse shared options from tag. Returns (tag, frozen_noise, from_pixels)."""
    frozen_noise = False
    while ":" in tag:
        x = tag.split(":")
        opt = x[-1]
        if opt == "fn":
            frozen_noise = True
        elif opt == "pixels":
            from_pixels = True if from_pixels is None else from_pixels
        else:
            break
        tag = ":".join(x[:-1])
    return tag, frozen_noise, from_pixels


def _atari_pong_policy(env_conf):
    bindings = _get_atari_dm_bindings()
    _env_id, policy_class = bindings.resolve_atari_from_tag("atari:Pong")
    return policy_class(env_conf)


def _gaussian_policy_factory(variant: str):
    from rl.policy_backbone import GaussianActorBackbonePolicyFactory

    return GaussianActorBackbonePolicyFactory(
        variant=variant,
        deterministic_eval=True,
        squash_mode="clip",
        init_log_std=-0.5,
    )


def get_env_conf(
    tag,
    problem_seed=None,
    noise_level=None,
    noise_seed_0=None,
    from_pixels=None,
    pixels_only=None,
    atari_preprocess=None,
):
    tag, frozen_noise, from_pixels = _parse_tag_options(tag, from_pixels)
    pix_only = pixels_only if pixels_only is not None else True

    if tag in _gym_env_confs:
        ec = copy.deepcopy(_gym_env_confs[tag])
    elif tag in _dm_control_env_confs:
        ec = copy.deepcopy(_dm_control_env_confs[tag])
    elif tag in _atari_env_confs:
        ec = copy.deepcopy(_atari_env_confs[tag])
    elif tag.startswith("dm:") or tag.startswith("dm_control/"):
        use_pixels = from_pixels if from_pixels is not None else False
        bindings = _get_atari_dm_bindings()
        env_name, policy_cls = bindings.resolve_dm_control_from_tag(tag, bool(use_pixels))
        ec = EnvConf(
            env_name,
            policy_class=policy_cls,
            from_pixels=use_pixels,
            pixels_only=pix_only,
            max_steps=_DM_CONTROL_DEFAULT_MAX_STEPS,
        )
    elif tag.startswith("atari:") or tag.startswith("ALE/"):
        bindings = _get_atari_dm_bindings()
        env_id, policy_cls = bindings.resolve_atari_from_tag(tag)
        ec = EnvConf(
            env_id,
            policy_class=policy_cls,
            from_pixels=True,
            pixels_only=True,
            max_steps=_ATARI_DEFAULT_MAX_STEPS,
        )
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
    rl_model: dict[str, Any] | None = None

    # dm_control pixel observations (RL and BO)
    from_pixels: bool = False
    pixels_only: bool = True
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
            make_dm_control_env = _get_atari_dm_bindings().make_dm_control_env
            env = make_dm_control_env(
                self.env_name,
                from_pixels=getattr(self, "from_pixels", False),
                pixels_only=getattr(self, "pixels_only", True),
                **kwargs,
            )
        elif self.env_name.startswith("ALE/"):
            bindings = _get_atari_dm_bindings()
            make_preprocess_options = bindings.make_atari_preprocess_options
            make_atari_env = bindings.make_atari_env
            render_mode = kwargs.get("render_mode")
            max_steps = self.max_steps
            if max_steps is None:
                raise ValueError("EnvConf.max_steps must be set for ALE environments.")
            preprocess = kwargs.pop("preprocess", None)
            if preprocess is None:
                default_preprocess = make_preprocess_options()
                preprocess_kwargs = asdict(default_preprocess) if is_dataclass(default_preprocess) else dict(vars(default_preprocess))
                if isinstance(self.atari_preprocess, dict):
                    preprocess_kwargs.update(self.atari_preprocess)
                preprocess = make_preprocess_options(**preprocess_kwargs)
            env = make_atari_env(
                self.env_name,
                render_mode=render_mode,
                max_episode_steps=int(max_steps),
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


def _gym_conf(env_name, gym_conf=None, policy_class=None, kwargs=None, noise_seed_0=None, rl_model=None):
    if gym_conf is None:
        gym_conf = GymConf()

    return EnvConf(
        env_name,
        gym_conf=gym_conf,
        policy_class=policy_class,
        kwargs=kwargs,
        noise_seed_0=noise_seed_0,
        rl_model=rl_model,
    )


def _normalize_rl_env_key(env_tag: str) -> str:
    tag, _frozen_noise, _from_pixels = _parse_tag_options(str(env_tag), None)
    if tag.startswith("dm:"):
        try:
            env_name, _policy_class = _get_atari_dm_bindings().resolve_dm_control_from_tag(tag, False)
            return str(env_name)
        except Exception:
            return tag
    if tag.startswith("atari:"):
        try:
            env_id, _policy_class = _get_atari_dm_bindings().resolve_atari_from_tag(tag)
            return str(env_id)
        except Exception:
            return tag
    return tag


def _find_rl_env_conf(env_key: str):
    registries = (_gym_env_confs, _dm_control_env_confs, _atari_env_confs)
    for registry in registries:
        direct = registry.get(env_key)
        if direct is not None:
            return direct
    for registry in registries:
        for conf in registry.values():
            if getattr(conf, "env_name", None) == env_key:
                return conf
    return None


def _infer_rl_from_policy_class(policy_class: Any, *, algo: str) -> dict[str, Any] | None:
    if not isinstance(policy_class, MLPPolicyFactory):
        return None
    hidden = tuple((int(v) for v in policy_class._hidden_sizes))
    layer_norm = bool(policy_class._use_layer_norm)
    if algo == "ppo":
        return {
            "backbone_name": "mlp",
            "backbone_hidden_sizes": hidden,
            "backbone_activation": "silu",
            "backbone_layer_norm": layer_norm,
            "actor_head_hidden_sizes": (),
            "critic_head_hidden_sizes": (),
            "head_activation": "silu",
            "share_backbone": True,
            "log_std_init": -0.5,
        }
    if algo == "sac":
        return {
            "backbone_name": "mlp",
            "backbone_hidden_sizes": hidden,
            "backbone_activation": "silu",
            "backbone_layer_norm": layer_norm,
            "actor_head_hidden_sizes": (),
            "critic_head_hidden_sizes": (),
            "head_activation": "silu",
        }
    raise ValueError(f"Unsupported algo '{algo}' for RL model inference.")


def resolve_rl_model_defaults(env_tag: str, *, algo: str) -> dict[str, Any]:
    algo_key = str(algo).strip().lower()
    if algo_key not in {"ppo", "sac"}:
        raise ValueError(f"Unsupported algo '{algo}'. Expected one of: ppo, sac.")
    env_key = _normalize_rl_env_key(str(env_tag))
    env_conf = _find_rl_env_conf(env_key)
    if env_conf is None:
        raise ValueError(f"No env preset found for env_tag '{env_tag}'. Add an entry to one of: _gym_env_confs, _dm_control_env_confs, _atari_env_confs.")
    base: dict[str, Any] = {}
    if env_conf.policy_class is not None:
        inferred = _infer_rl_from_policy_class(env_conf.policy_class, algo=algo_key)
        if inferred is not None:
            base = copy.deepcopy(inferred)
    explicit = None
    if isinstance(env_conf.rl_model, dict):
        model = env_conf.rl_model.get(algo_key)
        if isinstance(model, dict):
            explicit = copy.deepcopy(model)
    if explicit is not None:
        out = copy.deepcopy(base)
        out.update(explicit)
        return out
    if base:
        return copy.deepcopy(base)
    raise ValueError(
        f"No RL model defaults for env_tag '{env_tag}' and algo '{algo_key}'. "
        "Provide env_conf.rl_model[algo] or use an inferable MLPPolicyFactory policy_class."
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
        rl_model={
            "ppo": {
                "backbone_hidden_sizes": (64, 64),
                "backbone_layer_norm": True,
                "share_backbone": True,
                "log_std_init": -0.5,
            },
            "sac": {
                "backbone_hidden_sizes": (256, 256),
                "backbone_activation": "relu",
                "backbone_layer_norm": False,
                "head_activation": "relu",
            },
        },
    ),
    "cheetah-16x16": _gym_conf(
        "HalfCheetah-v5",
        policy_class=MLPPolicyFactory((16, 16)),
    ),
    "cheetah-16x16-gauss": _gym_conf(
        "HalfCheetah-v5",
        policy_class=_gaussian_policy_factory(variant="rl-gauss-tanh"),
    ),
    "cheetah-gauss": _gym_conf(
        "HalfCheetah-v5",
        policy_class=_gaussian_policy_factory(variant="rl-gauss-small"),
    ),
    "reach": EnvConf("Reacher-v5", gym_conf=GymConf(max_steps=50)),
    # "push": EnvConf("Pusher-v4",  gym_conf=GymConf(max_steps=100)),
    "hop": _gym_conf("Hopper-v5"),
    "hop-gauss": _gym_conf(
        "Hopper-v5",
        policy_class=_gaussian_policy_factory(variant="rl-gauss-small"),
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
    "dm_control/quadruped-run-v0": EnvConf(
        "dm_control/quadruped-run-v0",
        policy_class=MLPPolicyFactory((64, 64)),
        max_steps=_DM_CONTROL_DEFAULT_MAX_STEPS,
    ),
    "dm_control/quadruped-run-v0-small": EnvConf(
        "dm_control/quadruped-run-v0",
        policy_class=MLPPolicyFactory((4, 4)),
        max_steps=_DM_CONTROL_DEFAULT_MAX_STEPS,
    ),
}


_atari_env_confs = {
    "atari-pong": EnvConf(
        "ALE/Pong-v5",
        policy_class=_atari_pong_policy,
        from_pixels=True,
        pixels_only=True,
        max_steps=_ATARI_DEFAULT_MAX_STEPS,
        rl_model={
            "ppo": {
                "backbone_name": "nature_cnn_atari",
                "backbone_hidden_sizes": (),
                "backbone_activation": "relu",
                "backbone_layer_norm": False,
                "actor_head_hidden_sizes": (512,),
                "critic_head_hidden_sizes": (512,),
                "head_activation": "relu",
                "share_backbone": True,
                "log_std_init": -0.5,
            }
        },
    ),
}
