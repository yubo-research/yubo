from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol


class EnvironmentRuntimeProtocol(Protocol):
    """Protocol for environment runtime passed to policy factories.

    Matches the relevant fields from EnvConf that policies need for construction.
    """

    problem_seed: int | None
    env_name: str
    state_space: Any
    action_space: Any
    gym_conf: Any


if TYPE_CHECKING:
    from policies.policy_mixin import PolicyParamsMixin

    Policy = PolicyParamsMixin
else:
    Policy = Any


@dataclass
class PolicyPreset:
    """Policy preset with factory and optional RL algorithm defaults."""

    factory: Callable[[EnvironmentRuntimeProtocol], Policy]
    rl_model: dict[str, dict[str, Any]] | None = None


def _linear_factory(env_runtime: EnvironmentRuntimeProtocol) -> Policy:
    from problems.linear_policy import LinearPolicy

    return LinearPolicy(env_runtime)


def _pure_function_factory(env_runtime: EnvironmentRuntimeProtocol) -> Policy:
    from problems.pure_function_policy import PureFunctionPolicy

    return PureFunctionPolicy(env_runtime)


def _bipedal_heuristic_factory(env_runtime: EnvironmentRuntimeProtocol) -> Policy:
    from problems.bipedal_walker_policy import BipedalWalkerPolicy

    return BipedalWalkerPolicy(env_runtime)


def _turbo_lunar_factory(env_runtime: EnvironmentRuntimeProtocol) -> Policy:
    from problems.turbo_lunar_policy import TurboLunarPolicy

    return TurboLunarPolicy(env_runtime)


def _mlp_factory(
    hidden_sizes: tuple[int, ...],
) -> Callable[[EnvironmentRuntimeProtocol], Policy]:
    def factory(env_runtime: EnvironmentRuntimeProtocol) -> Policy:
        from policies.mlp_policy import MLPPolicyFactory

        return MLPPolicyFactory(hidden_sizes)(env_runtime)

    return factory


def _actor_critic_mlp_factory(
    hidden_sizes: tuple[int, ...],
) -> Callable[[EnvironmentRuntimeProtocol], Policy]:
    def factory(env_runtime: EnvironmentRuntimeProtocol) -> Policy:
        from policies.actor_critic_mlp_policy import ActorCriticMLPPolicyFactory

        return ActorCriticMLPPolicyFactory(hidden_sizes)(env_runtime)

    return factory


def _actor_mlp_factory(
    hidden_sizes: tuple[int, ...],
) -> Callable[[EnvironmentRuntimeProtocol], Policy]:
    def factory(env_runtime: EnvironmentRuntimeProtocol) -> Policy:
        from policies.actor_mlp_policy import ActorMLPPolicyFactory

        return ActorMLPPolicyFactory(hidden_sizes)(env_runtime)

    return factory


def _gaussian_backbone_factory(
    variant: str,
) -> Callable[[EnvironmentRuntimeProtocol], Policy]:
    from problems.env_conf_policies import gaussian_policy_factory

    def factory(env_runtime: EnvironmentRuntimeProtocol) -> Policy:
        return gaussian_policy_factory(variant)(env_runtime)

    return factory


def _atari_cnn_factory(env_runtime: EnvironmentRuntimeProtocol) -> Policy:
    from problems.pixel_policies import AtariCNNPolicyFactory

    return AtariCNNPolicyFactory(hidden_sizes=(512,), cnn_latent_dim=512, variant="default")(env_runtime)


def _infer_rl_model_from_mlp_like(hidden_sizes: tuple[int, ...], *, ppo_log_std_init: float) -> dict[str, dict[str, Any]]:
    return {
        "ppo": {
            "backbone_name": "mlp",
            "backbone_hidden_sizes": hidden_sizes,
            "backbone_activation": "silu",
            "backbone_layer_norm": True,
            "actor_head_hidden_sizes": (),
            "critic_head_hidden_sizes": (),
            "head_activation": "silu",
            "share_backbone": True,
            "log_std_init": ppo_log_std_init,
        },
        "sac": {
            "backbone_name": "mlp",
            "backbone_hidden_sizes": hidden_sizes,
            "backbone_activation": "silu",
            "backbone_layer_norm": True,
            "actor_head_hidden_sizes": (),
            "critic_head_hidden_sizes": (),
            "head_activation": "silu",
        },
    }


def _infer_rl_model_from_mlp(
    hidden_sizes: tuple[int, ...],
) -> dict[str, dict[str, Any]]:
    return _infer_rl_model_from_mlp_like(hidden_sizes, ppo_log_std_init=-0.5)


def _infer_rl_model_from_actor_critic_mlp(
    hidden_sizes: tuple[int, ...],
) -> dict[str, dict[str, Any]]:
    return _infer_rl_model_from_mlp_like(hidden_sizes, ppo_log_std_init=0.0)


def _infer_rl_model_from_actor_mlp(
    hidden_sizes: tuple[int, ...],
) -> dict[str, dict[str, Any]]:
    return {
        "ppo": {
            "backbone_name": "mlp",
            "backbone_hidden_sizes": hidden_sizes,
            "backbone_activation": "silu",
            "backbone_layer_norm": True,
            "actor_head_hidden_sizes": (),
            "head_activation": "silu",
            "log_std_init": 0.0,
        },
    }


def _infer_rl_model_from_atari_cnn() -> dict[str, dict[str, Any]]:
    return {
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
        },
    }


POLICY_PRESETS: dict[str, PolicyPreset] = {
    "linear": PolicyPreset(factory=_linear_factory),
    "pure-function": PolicyPreset(factory=_pure_function_factory),
    "mlp-16-8": PolicyPreset(
        factory=_mlp_factory((16, 8)),
        rl_model=_infer_rl_model_from_mlp((16, 8)),
    ),
    "mlp-16-16": PolicyPreset(
        factory=_mlp_factory((16, 16)),
        rl_model=_infer_rl_model_from_mlp((16, 16)),
    ),
    "mlp-32-16": PolicyPreset(
        factory=_mlp_factory((32, 16)),
        rl_model=_infer_rl_model_from_mlp((32, 16)),
    ),
    "mlp-64-64": PolicyPreset(
        factory=_mlp_factory((64, 64)),
        rl_model=_infer_rl_model_from_mlp((64, 64)),
    ),
    "mlp-256-128": PolicyPreset(
        factory=_mlp_factory((256, 128)),
        rl_model=_infer_rl_model_from_mlp((256, 128)),
    ),
    "mlp-1024-512-256-128": PolicyPreset(
        factory=_mlp_factory((1024, 512, 256, 128)),
        rl_model=_infer_rl_model_from_mlp((1024, 512, 256, 128)),
    ),
    "mlp-1024-600": PolicyPreset(
        factory=_mlp_factory((1024, 600)),
        rl_model=_infer_rl_model_from_mlp((1024, 600)),
    ),
    "mlp-4096-2060": PolicyPreset(
        factory=_mlp_factory((4096, 2060)),
        rl_model=_infer_rl_model_from_mlp((4096, 2060)),
    ),
    "mlp-32000-31000": PolicyPreset(
        factory=_mlp_factory((32000, 31000)),
        rl_model=_infer_rl_model_from_mlp((32000, 31000)),
    ),
    "mlp-4-4": PolicyPreset(
        factory=_mlp_factory((4, 4)),
        rl_model=_infer_rl_model_from_mlp((4, 4)),
    ),
    "bipedal-heuristic": PolicyPreset(factory=_bipedal_heuristic_factory),
    "turbo-lunar": PolicyPreset(factory=_turbo_lunar_factory),
    "actor-critic-mlp-16-8": PolicyPreset(
        factory=_actor_critic_mlp_factory((16, 8)),
        rl_model=_infer_rl_model_from_actor_critic_mlp((16, 8)),
    ),
    "actor-critic-mlp-32-32": PolicyPreset(
        factory=_actor_critic_mlp_factory((32, 32)),
        rl_model=_infer_rl_model_from_actor_critic_mlp((32, 32)),
    ),
    "actor-mlp-16-8": PolicyPreset(
        factory=_actor_mlp_factory((16, 8)),
        rl_model=_infer_rl_model_from_actor_mlp((16, 8)),
    ),
    "actor-mlp-32-32": PolicyPreset(
        factory=_actor_mlp_factory((32, 32)),
        rl_model=_infer_rl_model_from_actor_mlp((32, 32)),
    ),
    "gauss-rl-gauss-tanh": PolicyPreset(factory=_gaussian_backbone_factory("rl-gauss-tanh")),
    "gauss-rl-gauss-small": PolicyPreset(factory=_gaussian_backbone_factory("rl-gauss-small")),
    "atari-cnn": PolicyPreset(
        factory=_atari_cnn_factory,
        rl_model=_infer_rl_model_from_atari_cnn(),
    ),
}


def _parse_sizes_suffix(prefix: str, tag: str) -> tuple[int, ...] | None:
    if not tag.startswith(prefix):
        return None
    suffix = tag[len(prefix) :]
    if not suffix:
        return None
    sizes: list[int] = []
    for segment in suffix.split("-"):
        if not segment.isdigit() or int(segment) < 1:
            return None
        sizes.append(int(segment))
    return tuple(sizes)


def _preset_from_pattern(tag: str) -> PolicyPreset | None:
    for prefix, factory_fn, infer_fn in (
        ("actor-critic-mlp-", _actor_critic_mlp_factory, _infer_rl_model_from_actor_critic_mlp),
        ("actor-mlp-", _actor_mlp_factory, _infer_rl_model_from_actor_mlp),
        ("mlp-", _mlp_factory, _infer_rl_model_from_mlp),
    ):
        sizes = _parse_sizes_suffix(prefix, tag)
        if sizes is not None:
            return PolicyPreset(factory=factory_fn(sizes), rl_model=infer_fn(sizes))
    return None


def get_policy_preset(policy_tag: str) -> PolicyPreset:
    """Get a policy preset by tag.

    Raises KeyError if the tag is not found.
    """
    if policy_tag in POLICY_PRESETS:
        return POLICY_PRESETS[policy_tag]
    preset = _preset_from_pattern(policy_tag)
    if preset is not None:
        return preset
    raise KeyError(f"Unknown policy tag '{policy_tag}'. Available: {list_policy_tags()}")


def list_policy_tags() -> list[str]:
    """Return sorted list of available policy tags."""
    return sorted(POLICY_PRESETS.keys())
