"""Environment-only configuration (no policy_class, no rl_model).

This module provides environment specifications and runtime wrappers
separate from policy concerns.
"""

import copy
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable

import gymnasium as gym

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


def needs_atari_dm_bindings(env_tag: str) -> bool:
    """Return True if this env tag requires Atari/DM bindings to be registered."""
    tag, _, _ = parse_tag_options(str(env_tag), None)
    if tag.startswith(("dm:", "dm_control/", "atari:", "ALE/")):
        return True
    if tag in _dm_control_env_specs or tag in _atari_env_specs:
        return True
    if tag in _gym_env_specs:
        spec = _gym_env_specs[tag]
        return str(getattr(spec, "env_name", "")).startswith(("dm_control/", "ALE/"))
    return False


def parse_tag_options(tag, from_pixels):
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


@dataclass
class GymConf:
    max_steps: int = 1000
    num_frames_skip: int = 30
    state_space: Any = None
    transform_state: bool = True


@dataclass
class EnvironmentSpec:
    """Environment specification - NO policy_class, NO rl_model."""

    env_name: str
    gym_conf: GymConf | None = None
    kwargs: dict | None = None
    max_steps: int | None = None
    from_pixels: bool = False
    pixels_only: bool = True
    atari_preprocess: dict[str, Any] | None = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.env_name[:2] in ("f:", "g:") and self.max_steps is None:
            self.max_steps = _PURE_FUNCTION_MAX_STEPS
        if self.max_steps is None and self.gym_conf is None and not self.env_name.startswith(("ALE/", "dm_control/")):
            self.max_steps = _DEFAULT_MAX_STEPS


@dataclass
class EnvironmentRuntime:
    """Runtime environment with resolved spaces and make() capability."""

    spec: EnvironmentSpec
    problem_seed: int | None = None
    noise_level: float | None = None
    noise_seed_0: int | None = None
    frozen_noise: bool = True
    state_space: Any = None
    action_space: Any = None
    env_tag: str | None = None

    @property
    def env_name(self) -> str:
        return self.spec.env_name

    @property
    def gym_conf(self) -> GymConf | None:
        return self.spec.gym_conf

    @property
    def max_steps(self) -> int | None:
        if self.spec.gym_conf is not None:
            return self.spec.gym_conf.max_steps
        return self.spec.max_steps

    @property
    def kwargs(self) -> dict:
        return self.spec.kwargs or {}

    def _make(self, **kwargs):
        spec = self.spec
        env_name = spec.env_name

        if env_name[:2] == "f:":
            _ns: dict = {}
            exec("import problems.pure_functions as pf", _ns)  # noqa: S102
            env = _ns["pf"].make(env_name, problem_seed=self.problem_seed, distort=True)
        elif env_name[:2] == "g:":
            _ns: dict = {}
            exec("import problems.pure_functions as pf", _ns)  # noqa: S102
            env = _ns["pf"].make(env_name, problem_seed=self.problem_seed, distort=False)
        elif env_name.startswith("dm_control/"):
            make_dm_control_env = _get_atari_dm_bindings().make_dm_control_env
            env = make_dm_control_env(
                env_name,
                from_pixels=spec.from_pixels,
                pixels_only=spec.pixels_only,
                **kwargs,
            )
        elif env_name.startswith("ALE/"):
            bindings = _get_atari_dm_bindings()
            make_preprocess_options = bindings.make_atari_preprocess_options
            make_atari_env = bindings.make_atari_env
            render_mode = kwargs.get("render_mode")
            max_steps = spec.max_steps
            if max_steps is None:
                raise ValueError("EnvironmentSpec.max_steps must be set for ALE environments.")
            preprocess = kwargs.pop("preprocess", None)
            if preprocess is None:
                default_preprocess = make_preprocess_options()
                preprocess_kwargs = asdict(default_preprocess) if is_dataclass(default_preprocess) else dict(vars(default_preprocess))
                if isinstance(spec.atari_preprocess, dict):
                    preprocess_kwargs.update(spec.atari_preprocess)
                preprocess = make_preprocess_options(**preprocess_kwargs)
            env = make_atari_env(
                env_name,
                render_mode=render_mode,
                max_episode_steps=int(max_steps),
                preprocess=preprocess,
            )
        elif spec.gym_conf is not None:
            env = gym.make(env_name, **(kwargs | spec.kwargs))
        else:
            _ns: dict = {}
            exec("import problems.other as other", _ns)  # noqa: S102
            env = _ns["other"].make(env_name, problem_seed=self.problem_seed)

        return env

    def make(self, **kwargs):
        if self.spec.gym_conf:
            self.ensure_spaces()
        env = self._make(**kwargs)
        if self.noise_level is not None:
            assert self.spec.env_name[:2] in ["f:", "g:"], (
                "NYI: Noise is only supported for pure functions",
                self.spec.env_name,
            )
            _ns: dict = {}
            exec("from problems.noise_maker import NoiseMaker", _ns)  # noqa: S102
            env = _ns["NoiseMaker"](env, self.noise_level)
        return env

    def ensure_spaces(self):
        """Ensure state_space and action_space are populated (no-op if already set)."""
        if self.state_space is not None and self.action_space is not None:
            return
        spec = self.spec
        if spec.gym_conf is not None and spec.gym_conf.state_space is not None and self.action_space is not None:
            self.state_space = spec.gym_conf.state_space
            return
        env = self._make()
        self.state_space = env.observation_space
        if spec.gym_conf is not None:
            spec.gym_conf.state_space = self.state_space
        self.action_space = env.action_space
        env.close()


def _gym_spec(
    env_name: str,
    gym_conf: GymConf | None = None,
    kwargs: dict | None = None,
) -> EnvironmentSpec:
    if gym_conf is None:
        gym_conf = GymConf()
    return EnvironmentSpec(
        env_name,
        gym_conf=gym_conf,
        kwargs=kwargs,
    )


def get_environment_spec(env_tag: str) -> EnvironmentSpec:
    """Look up an environment spec by tag."""
    tag, _frozen_noise, from_pixels = parse_tag_options(str(env_tag), None)

    if tag in _gym_env_specs:
        return copy.deepcopy(_gym_env_specs[tag])
    if tag in _dm_control_env_specs:
        return copy.deepcopy(_dm_control_env_specs[tag])
    if tag in _atari_env_specs:
        return copy.deepcopy(_atari_env_specs[tag])

    pix_only = True
    if tag.startswith("dm:") or tag.startswith("dm_control/"):
        use_pixels = from_pixels if from_pixels is not None else False
        bindings = _get_atari_dm_bindings()
        env_name, _policy_cls = bindings.resolve_dm_control_from_tag(tag, bool(use_pixels))
        return EnvironmentSpec(
            env_name,
            from_pixels=use_pixels,
            pixels_only=pix_only,
            max_steps=_DM_CONTROL_DEFAULT_MAX_STEPS,
        )

    if tag.startswith("atari:") or tag.startswith("ALE/"):
        bindings = _get_atari_dm_bindings()
        env_id, _policy_cls = bindings.resolve_atari_from_tag(tag)
        return EnvironmentSpec(
            env_id,
            from_pixels=True,
            pixels_only=True,
            max_steps=_ATARI_DEFAULT_MAX_STEPS,
        )

    return EnvironmentSpec(tag)


def materialize_env(
    spec: EnvironmentSpec,
    *,
    problem_seed: int | None = None,
    noise_seed_0: int | None = None,
    noise_level: float | None = None,
    frozen_noise: bool = True,
    env_tag: str | None = None,
) -> EnvironmentRuntime:
    """Create an EnvironmentRuntime from a spec with resolved runtime parameters."""
    return EnvironmentRuntime(
        spec=spec,
        problem_seed=problem_seed,
        noise_level=noise_level,
        noise_seed_0=noise_seed_0,
        frozen_noise=frozen_noise,
        env_tag=env_tag,
    )


# Gym environment specs (stripped of policy_class and rl_model)
_gym_env_specs: dict[str, EnvironmentSpec] = {
    "mcc": _gym_spec(
        "MountainCarContinuous-v0",
        gym_conf=GymConf(num_frames_skip=100),
    ),
    "pend": EnvironmentSpec("Pendulum-v1", gym_conf=GymConf(max_steps=200, num_frames_skip=100)),
    "ant": _gym_spec("Ant-v5"),
    "mpend": _gym_spec("InvertedPendulum-v5"),
    "macro": _gym_spec("InvertedDoublePendulum-v5"),
    "swim": _gym_spec("Swimmer-v5"),
    "cheetah": _gym_spec("HalfCheetah-v5"),
    "quadruped-run-64x64": _gym_spec("dm_control/quadruped-run-v0"),
    "cheetah-16x16": _gym_spec("HalfCheetah-v5"),
    "cheetah-16x16-gauss": _gym_spec("HalfCheetah-v5"),
    "cheetah-gauss": _gym_spec("HalfCheetah-v5"),
    "reach": EnvironmentSpec("Reacher-v5", gym_conf=GymConf(max_steps=50)),
    "hop": _gym_spec("Hopper-v5"),
    "hop-gauss": _gym_spec("Hopper-v5"),
    "human": _gym_spec("Humanoid-v5"),
    "stand": _gym_spec("HumanoidStandup-v5"),
    "stand-mlp": _gym_spec("HumanoidStandup-v5"),
    "stand-mlp2": _gym_spec("HumanoidStandup-v5"),
    "stand-mlp3": _gym_spec("HumanoidStandup-v5"),
    "stand-mlp4": _gym_spec("HumanoidStandup-v5"),
    "stand-mlp5": _gym_spec("HumanoidStandup-v5"),
    "bw": _gym_spec(
        "BipedalWalker-v3",
        gym_conf=GymConf(
            max_steps=1600,
            num_frames_skip=100,
        ),
    ),
    "bw-linraw": _gym_spec(
        "BipedalWalker-v3",
        gym_conf=GymConf(
            max_steps=1600,
            num_frames_skip=100,
            transform_state=False,
        ),
    ),
    "bw-mlp": _gym_spec(
        "BipedalWalker-v3",
        gym_conf=GymConf(
            max_steps=1600,
            num_frames_skip=100,
        ),
    ),
    "bw-heur": _gym_spec(
        "BipedalWalker-v3",
        gym_conf=GymConf(
            max_steps=1600,
            num_frames_skip=100,
            transform_state=False,
        ),
    ),
    "lunar": _gym_spec(
        "LunarLander-v3",
        gym_conf=GymConf(
            max_steps=500,
        ),
        kwargs={"continuous": True},
    ),
    "lunar-mlp": _gym_spec(
        "LunarLander-v3",
        gym_conf=GymConf(
            max_steps=500,
        ),
        kwargs={"continuous": True},
    ),
    "lunar-ac": _gym_spec(
        "LunarLander-v3",
        gym_conf=GymConf(
            max_steps=500,
        ),
        kwargs={"continuous": True},
    ),
    "tlunar": EnvironmentSpec(
        "LunarLander-v3",
        gym_conf=GymConf(
            max_steps=500,
            transform_state=False,
        ),
        kwargs={"continuous": False},
    ),
}


# DM Control environment specs (stripped of policy_class and rl_model)
_dm_control_env_specs: dict[str, EnvironmentSpec] = {
    "dm_control/quadruped-run-v0": EnvironmentSpec(
        "dm_control/quadruped-run-v0",
        max_steps=_DM_CONTROL_DEFAULT_MAX_STEPS,
    ),
    "dm_control/quadruped-run-v0-small": EnvironmentSpec(
        "dm_control/quadruped-run-v0",
        max_steps=_DM_CONTROL_DEFAULT_MAX_STEPS,
    ),
}


# Atari environment specs (stripped of policy_class and rl_model)
_atari_env_specs: dict[str, EnvironmentSpec] = {
    "atari-pong": EnvironmentSpec(
        "ALE/Pong-v5",
        from_pixels=True,
        pixels_only=True,
        max_steps=_ATARI_DEFAULT_MAX_STEPS,
    ),
}
