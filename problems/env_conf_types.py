from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

import gymnasium as gym
import numpy as np

import problems.other as other
import problems.pure_functions as pure_functions
from policies.mlp_policy import MLPPolicy
from problems.env_conf_bindings import get_atari_dm_bindings
from problems.env_conf_constants import _DEFAULT_MAX_STEPS, _PURE_FUNCTION_MAX_STEPS
from problems.isaaclab_env_adapters import (
    is_isaaclab_env_tag,
    make_isaaclab_env,
    resolve_isaaclab_env_spaces,
)
from problems.mlp_torch_env import wrap_mlp_env
from problems.noise_maker import NoiseMaker


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
        if self.env_name.startswith("warp:"):
            from problems.unified_mj_env import UnifiedMJXWarpAdapter

            num_envs = kwargs.get("num_envs", 1)
            return UnifiedMJXWarpAdapter(self.env_name[5:], backend="warp", num_envs=num_envs)
        if self.env_name[:2] in ("f:", "g:"):
            env = pure_functions.make(self.env_name, problem_seed=self.problem_seed, distort=self.env_name[:2] == "f:")
        elif self.env_name.startswith("dm_control/"):
            make_dm_control_env = get_atari_dm_bindings().make_dm_control_env
            env = make_dm_control_env(
                self.env_name,
                from_pixels=getattr(self, "from_pixels", False),
                pixels_only=getattr(self, "pixels_only", True),
                **kwargs,
            )
        elif self.env_name.startswith("ALE/"):
            bindings = get_atari_dm_bindings()
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
        elif is_isaaclab_env_tag(self.env_name):
            env = make_isaaclab_env(self.env_name, seed=self.problem_seed, **(kwargs | self.kwargs))
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

    def make_gym_env(self, *, seed: int | None = None, render_mode: str | None = "rgb_array", **kwargs) -> gym.Env:
        """Unified creation path for Gymnasium-compatible environments.

        Handles:
        1. Base environment creation (via _make) with seeding
        2. Pixel formatting (Resize, Grayscale, Transpose)
        3. Frame stacking (if specified)
        4. Action clipping (if Box action space)
        5. Standard preprocessing (via common.env_preprocessing)
        """
        from gymnasium.wrappers import (
            ClipAction,
            FrameStackObservation,
            GrayscaleObservation,
            ResizeObservation,
            TransformObservation,
        )

        import common.env_preprocessing as env_pre

        env_kwargs = (self.kwargs or {}).copy()
        env_kwargs.update(kwargs)

        # 1. Base creation
        try:
            env = self._make(render_mode=render_mode, **env_kwargs)
        except TypeError:
            env = self._make(**env_kwargs)

        if seed is not None:
            env.reset(seed=seed)
            if hasattr(env.action_space, "seed"):
                env.action_space.seed(seed)

        # 2. Pixel mode processing (if from_pixels=True)
        if getattr(self, "from_pixels", False):
            # Atari (ALE/*) environments already handle resizing/grayscale if configured via atari_preprocess.
            # For others (DM Control, Generic Gym), we apply standard wrappers.
            if not self.env_name.startswith("ALE/"):
                # If it's a Dict obs (like DM Control), we might need to select the 'pixels' key first
                # (but our DM Control wrapper already returns pixels as the main obs if pixels_only=True)

                # Resize to standard 84x84
                env = ResizeObservation(env, (84, 84))

                # Optional Grayscale (Atari usually wants this, but DM Control usually keeps RGB)
                if "atari" in self.env_name.lower():
                    env = GrayscaleObservation(env, keep_dim=True)

                # Transpose to (C, H, W) for PyTorch/NatureCNN compatibility
                # (standard Gym ResizeObservation returns (H, W, C))
                env = TransformObservation(env, lambda o: np.moveaxis(o, -1, -3))

        # 3. Frame skip/stack
        # Note: ALEAtariEnv handles skip natively.
        if self.gym_conf and self.gym_conf.num_frames_skip > 1:
            # For RL, we usually stack frames if skipping
            num_stack = 4 if "atari" in self.env_name.lower() else 1
            if num_stack > 1:
                env = FrameStackObservation(env, stack_size=num_stack)

        # 4. Action clipping
        if isinstance(env.action_space, gym.spaces.Box):
            env = ClipAction(env)

        # 5. Preprocessing (Normalization, Clipping)
        env = env_pre.apply_gym_preprocessing(env, env_conf=self)

        return env

    def ensure_spaces(self):
        """Ensure state_space and action_space are populated (no-op if already set)."""
        if self.state_space is not None and self.action_space is not None:
            return
        if self.gym_conf is not None and self.gym_conf.state_space is not None and self.action_space is not None:
            self.state_space = self.gym_conf.state_space
            return
        if is_isaaclab_env_tag(self.env_name):
            self.state_space, self.action_space = resolve_isaaclab_env_spaces(
                self.env_name,
                launcher_kwargs=self.kwargs.get("launcher_kwargs"),
            )
            if self.gym_conf is not None:
                self.gym_conf.state_space = self.state_space
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
        is_deferred = self.env_name.startswith(("ALE/", "dm_control/", "warp:")) or is_isaaclab_env_tag(self.env_name)
        if self.max_steps is None and self.gym_conf is None and not is_deferred:
            self.max_steps = _DEFAULT_MAX_STEPS
        if self.gym_conf:
            # Defer gym.make to avoid eagerly instantiating all envs at import time.
            self.gym_conf.state_space = None
            self.state_space = None
            self.action_space = None
            return
        if is_deferred:
            # Defer heavy env creation for Atari/DM/Isaac Lab until first use.
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


def _gym_conf(
    env_name,
    gym_conf=None,
    policy_class=None,
    kwargs=None,
    noise_seed_0=None,
    rl_model=None,
):
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
