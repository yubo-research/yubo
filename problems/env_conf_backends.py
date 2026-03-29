from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

from problems.env_conf_policies import (
    resolve_atari_policy_class,
    resolve_dm_control_policy_class,
)

_DM_POLICY_VARIANTS = frozenset({"gauss", "rl-gauss"})
_ATARI_POLICY_VARIANTS = frozenset({"agent57", "gauss", "mlp16"})


@dataclass(frozen=True)
class AtariDMBindings:
    # Tag -> (env_name, policy_class)
    resolve_dm_control_from_tag: Callable[[str, bool], tuple[str, Any]]
    resolve_atari_from_tag: Callable[[str], tuple[str, Any]]
    # Atari preprocess/options
    make_atari_preprocess_options: Callable[..., Any]
    # Environment constructors
    make_dm_control_env: Callable[..., Any]
    make_atari_env: Callable[..., Any]


def load_atari_dm_bindings() -> AtariDMBindings:
    atari_module: Any | None = None
    dm_make: Callable[..., Any] | None = None

    def _get_atari_module():
        nonlocal atari_module
        if atari_module is None:
            atari_module = importlib.import_module("problems.atari_env")
        return atari_module

    def _make_atari_env(*args, **kwargs):
        module = _get_atari_module()
        return module.make(*args, **kwargs)

    def _make_atari_preprocess_options(**kwargs):
        module = _get_atari_module()
        return module.AtariPreprocessOptions(**kwargs)

    def _make_dm_control_env(*args, **kwargs):
        nonlocal dm_make
        if dm_make is None:
            from problems.dm_control_env import make as make_dm_control_env

            dm_make = make_dm_control_env
        return dm_make(*args, **kwargs)

    def _split_variant_suffix(tag: str, allowed_variants: set[str]) -> tuple[str, str | None]:
        parts = str(tag).split(":")
        if len(parts) >= 2 and parts[-1] in allowed_variants:
            return (":".join(parts[:-1]), parts[-1])
        return (str(tag), None)

    def _dm_base_and_variant(tag: str) -> tuple[str, str | None]:
        return _split_variant_suffix(tag, _DM_POLICY_VARIANTS)

    def _atari_base_and_variant(tag: str) -> tuple[str, str | None]:
        return _split_variant_suffix(tag, _ATARI_POLICY_VARIANTS)

    def _normalize_dm_control_name(tag: str) -> str:
        if tag.startswith("dm:"):
            name = tag.split(":", 1)[1]
        else:
            name = tag.split("/", 1)[1]
        if not name.endswith("-v0") and not name.endswith("-v1"):
            name = f"{name}-v0"
        return f"dm_control/{name}"

    def _resolve_dm_control_from_tag(tag: str, use_pixels: bool) -> tuple[str, Any]:
        from problems.pixel_policies import CNNMLPPolicyFactory

        base_tag, policy_variant = _dm_base_and_variant(tag)
        env_name = _normalize_dm_control_name(base_tag)
        policy_class = resolve_dm_control_policy_class(
            use_pixels=bool(use_pixels),
            policy_variant=policy_variant,
            cnn_mlp_policy_factory=CNNMLPPolicyFactory,
        )
        return (env_name, policy_class)

    def _resolve_atari_from_tag(tag: str) -> tuple[str, Any]:
        from problems.pixel_policies import (
            AtariAgent57LiteFactory,
            AtariCNNPolicyFactory,
            AtariGaussianPolicyFactory,
        )

        base_tag, policy_variant = _atari_base_and_variant(tag)
        module = _get_atari_module()
        env_id = module._parse_atari_tag(base_tag)
        policy_class = resolve_atari_policy_class(
            policy_variant=policy_variant,
            atari_agent57_factory=AtariAgent57LiteFactory,
            atari_cnn_policy_factory=AtariCNNPolicyFactory,
            atari_gaussian_policy_factory=AtariGaussianPolicyFactory,
        )
        return (env_id, policy_class)

    return AtariDMBindings(
        resolve_dm_control_from_tag=_resolve_dm_control_from_tag,
        resolve_atari_from_tag=_resolve_atari_from_tag,
        make_atari_preprocess_options=_make_atari_preprocess_options,
        make_dm_control_env=_make_dm_control_env,
        make_atari_env=_make_atari_env,
    )


def register_with_env_conf() -> None:
    from problems.env_conf import register_atari_dm_bindings_loader

    register_atari_dm_bindings_loader(load_atari_dm_bindings)
