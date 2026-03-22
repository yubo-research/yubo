from __future__ import annotations

from typing import TYPE_CHECKING, Any

from problems.environment_spec import (
    EnvironmentRuntime,
    get_environment_spec,
    materialize_env,
    parse_tag_options,
)

if TYPE_CHECKING:
    from policies.registry import PolicyPreset


def _get_policy_preset(policy_tag: str):
    _ns: dict = {}
    exec("from policies.registry import get_policy_preset", _ns)  # noqa: S102
    return _ns["get_policy_preset"](policy_tag)


_ENV_TAG_TO_POLICY_TAG: dict[str, str] = {
    "cheetah": "mlp-32-16",
    "cheetah-16x16": "mlp-16-16",
    "cheetah-16x16-gauss": "gauss-rl-gauss-tanh",
    "cheetah-gauss": "gauss-rl-gauss-small",
    "stand-mlp": "mlp-32-16",
    "stand-mlp2": "mlp-256-128",
    "stand-mlp3": "mlp-1024-600",
    "stand-mlp4": "mlp-4096-2060",
    "stand-mlp5": "mlp-32000-31000",
    "bw-mlp": "mlp-1024-512-256-128",
    "bw-heur": "bipedal-heuristic",
    "lunar-mlp": "mlp-16-8",
    "lunar-ac": "actor-critic-mlp-16-8",
    "tlunar": "turbo-lunar",
    "hop-gauss": "gauss-rl-gauss-small",
    "quadruped-run-64x64": "mlp-64-64",
    "dm_control/quadruped-run-v0": "mlp-64-64",
    "dm_control/quadruped-run-v0-small": "mlp-4-4",
    "atari-pong": "atari-cnn",
}

_PROBLEM_RL_MODEL_OVERRIDES: dict[tuple[str, str, str], dict[str, Any]] = {
    ("cheetah", "mlp-32-16", "ppo"): {
        "backbone_hidden_sizes": (64, 64),
        "backbone_layer_norm": True,
        "share_backbone": True,
        "log_std_init": -0.5,
    },
    ("cheetah", "mlp-32-16", "sac"): {
        "backbone_hidden_sizes": (256, 256),
        "backbone_activation": "relu",
        "backbone_layer_norm": False,
        "head_activation": "relu",
    },
    ("dm_control/quadruped-run-v0", "mlp-64-64", "ppo"): {
        "backbone_hidden_sizes": (64, 64),
        "backbone_layer_norm": True,
        "share_backbone": True,
        "log_std_init": -0.5,
    },
    ("dm_control/quadruped-run-v0", "mlp-64-64", "sac"): {
        "backbone_hidden_sizes": (256, 256),
        "backbone_activation": "relu",
        "backbone_layer_norm": True,
        "head_activation": "relu",
    },
    ("atari-pong", "atari-cnn", "ppo"): {
        "backbone_name": "nature_cnn_atari",
        "backbone_hidden_sizes": (),
        "backbone_activation": "relu",
        "backbone_layer_norm": False,
        "actor_head_hidden_sizes": (512,),
        "critic_head_hidden_sizes": (512,),
        "head_activation": "relu",
        "share_backbone": True,
        "log_std_init": -0.5,
    },
}


class Problem:
    """Bundles an environment runtime with a policy tag for lazy policy construction."""

    def __init__(self, env: EnvironmentRuntime, policy_tag: str) -> None:
        self._env = env
        self._policy_tag = policy_tag

    @property
    def env(self) -> EnvironmentRuntime:
        return self._env

    @property
    def policy_tag(self) -> str:
        return self._policy_tag

    def build_policy(self) -> Any:
        """Build and return the policy, ensuring spaces are populated first."""
        self._env.ensure_spaces()
        preset: PolicyPreset = _get_policy_preset(self._policy_tag)
        return preset.factory(self._env)


def infer_default_policy_tag(env_tag: str) -> str:
    """Infer a default policy tag based on environment type."""
    tag, _, _ = parse_tag_options(env_tag, None)
    if tag.startswith(("f:", "g:")):
        return "pure-function"
    if tag in _ENV_TAG_TO_POLICY_TAG:
        return _ENV_TAG_TO_POLICY_TAG[tag]
    return "linear"


def resolve_rl_model_defaults(
    env_tag: str,
    policy_tag: str | None = None,
    *,
    algo: str,
) -> dict[str, Any]:
    """Resolve RL model defaults for an (env_tag, policy_tag, algo) combination.

    Precedence:
        1. Problem-level override for (env_tag, policy_tag, algo)
        2. PolicyPreset.rl_model[algo]
        3. Error (no default available)
    """
    import copy

    algo_key = str(algo).strip().lower()
    if algo_key not in {"ppo", "sac"}:
        raise ValueError(f"Unsupported algo '{algo}'. Expected one of: ppo, sac.")

    tag, _, _ = parse_tag_options(env_tag, None)
    if policy_tag is None:
        policy_tag = infer_default_policy_tag(env_tag)

    override_key = (tag, policy_tag, algo_key)
    if override_key in _PROBLEM_RL_MODEL_OVERRIDES:
        return copy.deepcopy(_PROBLEM_RL_MODEL_OVERRIDES[override_key])

    preset = _get_policy_preset(policy_tag)
    if preset.rl_model is not None and algo_key in preset.rl_model:
        return copy.deepcopy(preset.rl_model[algo_key])

    raise ValueError(
        f"No RL model defaults for env_tag '{env_tag}', policy_tag '{policy_tag}', algo '{algo_key}'. Provide Problem-level override or PolicyPreset.rl_model."
    )


def build_problem(
    env_tag: str,
    policy_tag: str | None = None,
    *,
    problem_seed: int | None = None,
    noise_seed_0: int | None = None,
    noise_level: float | None = None,
    frozen_noise: bool = True,
    from_pixels: bool | None = None,
    pixels_only: bool | None = None,
) -> Problem:
    """Build a Problem from env and policy tags.

    Args:
        env_tag: Environment tag (e.g. "cheetah", "f:sphere", "dm:walker-walk").
        policy_tag: Policy tag (e.g. "mlp-32-16", "pure-function"). If None,
            infers a default based on env_tag.
        problem_seed: Seed for problem randomization (fixed per optimization run).
        noise_seed_0: Initial noise seed for denoising runs.
        noise_level: Optional noise level for pure function environments.
        frozen_noise: Whether noise seeds are frozen across rounds.
        from_pixels: Override spec's from_pixels setting for pixel observations.
        pixels_only: Override spec's pixels_only setting.

    Returns:
        A Problem instance with lazy policy construction.
    """
    if policy_tag is None:
        policy_tag = infer_default_policy_tag(env_tag)

    spec = get_environment_spec(env_tag)

    if from_pixels is not None:
        spec.from_pixels = from_pixels
    if pixels_only is not None:
        spec.pixels_only = pixels_only

    runtime = materialize_env(
        spec,
        problem_seed=problem_seed,
        noise_seed_0=noise_seed_0,
        noise_level=noise_level,
        frozen_noise=frozen_noise,
        env_tag=env_tag,
    )
    return Problem(runtime, policy_tag)
