from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _tuple_ints(values) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def _schema(factory: Any) -> dict[str, Any]:
    to_rl_schema = getattr(factory, "to_rl_schema", None)
    if to_rl_schema is not None:
        return dict(to_rl_schema())
    hidden_sizes = getattr(factory, "hidden_sizes", None)
    if hidden_sizes is None:
        raise ValueError("policy_factory must define to_rl_schema() or expose hidden_sizes.")
    return {
        "family": "mlp",
        "backbone_hidden_sizes": _tuple_ints(hidden_sizes),
        "backbone_activation": str("silu" if getattr(factory, "activation", None) is None else factory.activation),
        "backbone_layer_norm": bool(getattr(factory, "use_layer_norm", False)),
    }


def _mlp_backbone_fields(factory: Any, *, default_activation: str, default_layer_norm: bool) -> dict[str, Any]:
    schema = _schema(factory)
    if str(schema.get("family", "")).strip().lower() != "mlp":
        raise ValueError("RL scaffold critic/head factories must currently map to the 'mlp' family.")
    return {
        "backbone_name": "mlp",
        "backbone_hidden_sizes": _tuple_ints(schema.get("backbone_hidden_sizes", ())),
        "backbone_activation": str(schema.get("backbone_activation", default_activation)),
        "backbone_layer_norm": bool(schema.get("backbone_layer_norm", default_layer_norm)),
    }


def _mlp_head_fields(factory: Any, *, default_activation: str) -> dict[str, Any]:
    schema = _schema(factory)
    if str(schema.get("family", "")).strip().lower() != "mlp":
        raise ValueError("RL scaffold critic/head factories must currently map to the 'mlp' family.")
    return {
        "hidden_sizes": _tuple_ints(schema.get("backbone_hidden_sizes", ())),
        "activation": str(schema.get("backbone_activation", default_activation)),
    }


def _standard_model(
    schema: dict[str, Any],
    *,
    backbone_name: str,
    backbone_hidden_sizes,
    backbone_activation: str,
    head_activation: str,
) -> dict[str, Any]:
    model = {
        "backbone_name": str(backbone_name),
        "backbone_hidden_sizes": _tuple_ints(backbone_hidden_sizes),
        "backbone_activation": str(backbone_activation),
        "backbone_layer_norm": bool(schema.get("backbone_layer_norm", False)),
        "actor_head_hidden_sizes": _tuple_ints(schema.get("actor_head_hidden_sizes", ())),
        "critic_head_hidden_sizes": _tuple_ints(schema.get("critic_head_hidden_sizes", ())),
        "head_activation": str(head_activation),
    }
    if "actor_head_activation" in schema:
        model["actor_head_activation"] = str(schema["actor_head_activation"])
    if "critic_head_activation" in schema:
        model["critic_head_activation"] = str(schema["critic_head_activation"])
    if "critic_backbone_name" in schema:
        model["critic_backbone_name"] = str(schema["critic_backbone_name"])
        model["critic_backbone_hidden_sizes"] = _tuple_ints(schema.get("critic_backbone_hidden_sizes", ()))
        model["critic_backbone_activation"] = str(schema.get("critic_backbone_activation", backbone_activation))
        model["critic_backbone_layer_norm"] = bool(schema.get("critic_backbone_layer_norm", False))
    return model


def _mlp_model(schema: dict[str, Any]) -> dict[str, Any]:
    if schema.get("rnn_hidden_size") is not None:
        raise ValueError("RNN MLP schema does not map to the current RL model config.")
    if bool(schema.get("use_prev_action", False)):
        raise ValueError("Prev-action MLP schema does not map to the current RL model config.")
    if bool(schema.get("use_phase_features", False)):
        raise ValueError("Phase-feature MLP schema does not map to the current RL model config.")
    backbone_activation = str(schema.get("backbone_activation", "silu"))
    return _standard_model(
        schema,
        backbone_name="mlp",
        backbone_hidden_sizes=schema.get("backbone_hidden_sizes", ()),
        backbone_activation=backbone_activation,
        head_activation=str(schema.get("head_activation", backbone_activation)),
    )


def _cnn_model(schema: dict[str, Any], *, family: str) -> dict[str, Any]:
    if family == "nature_cnn_atari" and str(schema.get("variant", "default")).strip().lower() != "default":
        raise ValueError(f"Atari CNN variant '{schema.get('variant')}' does not map to the current RL model config.")
    backbone_activation = str(schema.get("backbone_activation", "relu"))
    return _standard_model(
        schema,
        backbone_name=family,
        backbone_hidden_sizes=(),
        backbone_activation=backbone_activation,
        head_activation=str(schema.get("head_activation", "relu" if family == "nature_cnn_atari" else "silu")),
    )


def _gaussian_backbone_model(schema: dict[str, Any]) -> dict[str, Any]:
    from rl.shared_gaussian_actor import get_gaussian_actor_spec

    backbone_spec, head_spec = get_gaussian_actor_spec(str(schema.get("variant", "rl-gauss-tanh")))
    return _standard_model(
        schema,
        backbone_name=str(backbone_spec.name),
        backbone_hidden_sizes=backbone_spec.hidden_sizes,
        backbone_activation=str(backbone_spec.activation),
        head_activation=str(head_spec.activation),
    ) | {
        "actor_head_hidden_sizes": _tuple_ints(head_spec.hidden_sizes),
        "critic_head_hidden_sizes": _tuple_ints(head_spec.hidden_sizes),
        "backbone_layer_norm": bool(backbone_spec.layer_norm),
    }


def project(schema: dict[str, Any], algo: str) -> dict[str, Any]:
    algo_key = str(algo).strip().lower()
    if algo_key not in {"ppo", "sac"}:
        raise ValueError(f"Unsupported algo '{algo}'. Expected one of: ppo, sac.")
    family = str(schema.get("family", "")).strip().lower()
    builders = {
        "mlp": _mlp_model,
        "nature_cnn": lambda s: _cnn_model(s, family="nature_cnn"),
        "nature_cnn_atari": lambda s: _cnn_model(s, family="nature_cnn_atari"),
        "gaussian_backbone": _gaussian_backbone_model,
    }
    if family not in builders:
        raise ValueError(f"Unsupported RL schema family '{family}'.")
    model = builders[family](schema)
    if algo_key == "ppo":
        model["share_backbone"] = bool(schema.get("share_backbone", True))
        model["log_std_init"] = float(schema.get("log_std_init", -0.5))
    overrides = schema.get(f"{algo_key}_overrides")
    if overrides is not None:
        model.update(dict(overrides))
    return model


@dataclass(frozen=True)
class RLPolicyFactory:
    policy_factory: Any
    critic: Any | None = None
    actor_head: Any | None = None
    critic_head: Any | None = None
    share_backbone: bool | None = None
    log_std_init: float | None = None
    ppo_overrides: dict[str, Any] | None = None
    sac_overrides: dict[str, Any] | None = None

    def __call__(self, env_conf):
        return self.policy_factory(env_conf)

    def to_rl_schema(self) -> dict[str, Any]:
        schema = _schema(self.policy_factory)
        default_head_activation = str(schema.get("head_activation", schema.get("backbone_activation", "silu")))
        if self.actor_head is not None:
            actor_head = _mlp_head_fields(self.actor_head, default_activation=default_head_activation)
            schema["actor_head_hidden_sizes"] = actor_head["hidden_sizes"]
            schema["actor_head_activation"] = actor_head["activation"]
        if self.critic_head is not None:
            critic_head = _mlp_head_fields(self.critic_head, default_activation=default_head_activation)
            schema["critic_head_hidden_sizes"] = critic_head["hidden_sizes"]
            schema["critic_head_activation"] = critic_head["activation"]
        if self.critic is not None:
            critic = _mlp_backbone_fields(
                self.critic,
                default_activation=str(schema.get("backbone_activation", "silu")),
                default_layer_norm=bool(schema.get("backbone_layer_norm", False)),
            )
            schema.update({f"critic_{k}": v for k, v in critic.items() if k != "backbone_name"})
            schema["critic_backbone_name"] = critic["backbone_name"]
            if self.share_backbone is None:
                schema["share_backbone"] = False
        if self.share_backbone is not None:
            schema["share_backbone"] = bool(self.share_backbone)
        if self.log_std_init is not None:
            schema["log_std_init"] = float(self.log_std_init)
        if self.ppo_overrides:
            schema["ppo_overrides"] = dict(self.ppo_overrides)
        if self.sac_overrides:
            schema["sac_overrides"] = dict(self.sac_overrides)
        return schema


def gaussian_policy_factory(variant: str, **kwargs: Any):
    from rl.policy_backbone import GaussianActorBackbonePolicyFactory

    return GaussianActorBackbonePolicyFactory(
        variant=variant,
        deterministic_eval=True,
        squash_mode="clip",
        init_log_std=-0.5,
        **dict(kwargs),
    )
