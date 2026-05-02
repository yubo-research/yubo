from __future__ import annotations

from typing import Any

from policies.mlp_policy import MLPPolicyFactory


def gaussian_policy_factory(variant: str, **kwargs: Any):
    _ns: dict = {}
    exec("from rl.policy_backbone import GaussianActorBackbonePolicyFactory", _ns)  # noqa: S102
    GaussianActorBackbonePolicyFactory = _ns["GaussianActorBackbonePolicyFactory"]
    return GaussianActorBackbonePolicyFactory(
        variant=variant,
        deterministic_eval=True,
        squash_mode="clip",
        init_log_std=-0.5,
        **dict(kwargs),
    )


def resolve_dm_control_policy_class(
    *,
    use_pixels: bool,
    policy_variant: str | None,
    cnn_mlp_policy_factory: Any = None,
):
    if use_pixels:
        if cnn_mlp_policy_factory is None:
            raise ValueError("cnn_mlp_policy_factory is required for dm_control pixel policies.")
        return cnn_mlp_policy_factory((32, 16))
    if policy_variant == "gauss":
        return gaussian_policy_factory(variant="rl-gauss-tanh")
    if policy_variant == "rl-gauss":
        return gaussian_policy_factory(variant="rl-gauss")
    return MLPPolicyFactory((32, 16))


def resolve_atari_policy_class(
    *,
    policy_variant: str | None,
    atari_agent57_factory: Any = None,
    atari_cnn_policy_factory: Any = None,
    atari_gaussian_policy_factory: Any = None,
):
    if policy_variant == "mlp16":
        _ns: dict = {}
        exec("from rl.policy_backbone import AtariMLP16DiscretePolicy", _ns)  # noqa: S102
        return _ns["AtariMLP16DiscretePolicy"]
    if policy_variant == "agent57":
        if atari_agent57_factory is None:
            raise ValueError("atari_agent57_factory is required for policy_variant='agent57'.")
        return atari_agent57_factory(lstm_hidden=32, cnn_variant="small")
    if policy_variant == "gauss":
        if atari_gaussian_policy_factory is None:
            raise ValueError("atari_gaussian_policy_factory is required for policy_variant='gauss'.")
        return atari_gaussian_policy_factory(
            hidden_sizes=(16, 16),
            cnn_latent_dim=64,
            variant="small",
            deterministic_eval=True,
            init_log_std=-0.5,
        )
    if atari_cnn_policy_factory is None:
        raise ValueError("atari_cnn_policy_factory is required for default Atari policy.")
    return atari_cnn_policy_factory((24,), variant="small")
