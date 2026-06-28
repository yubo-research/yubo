from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any

ROLE_ATTENTION_Q = "attention_q"
ROLE_ATTENTION_K = "attention_k"
ROLE_ATTENTION_V = "attention_v"
ROLE_ATTENTION_QKV = "attention_qkv"
ROLE_ATTENTION_O = "attention_o"
ROLE_MLP_GATE = "mlp_gate"
ROLE_MLP_UP = "mlp_up"
ROLE_MLP_DOWN = "mlp_down"
ROLE_MOE_ROUTER = "moe_router"
ROLE_MOE_EXPERT_GATE = "moe_expert_gate"
ROLE_MOE_EXPERT_UP = "moe_expert_up"
ROLE_MOE_EXPERT_DOWN = "moe_expert_down"
ROLE_MOE_SHARED_GATE = "moe_shared_gate"
ROLE_MOE_SHARED_UP = "moe_shared_up"
ROLE_MOE_SHARED_DOWN = "moe_shared_down"
ROLE_SSM_IN = "ssm_in"
ROLE_SSM_DT = "ssm_dt"
ROLE_SSM_CONV = "ssm_conv"
ROLE_SSM_OUT = "ssm_out"
ROLE_RNN_MIXER = "rnn_mixer"
ROLE_EMBEDDING = "embedding"
ROLE_NORM = "norm"
ROLE_LM_HEAD = "lm_head"
ROLE_OTHER = "other"

CANONICAL_ROLES = (
    ROLE_ATTENTION_Q,
    ROLE_ATTENTION_K,
    ROLE_ATTENTION_V,
    ROLE_ATTENTION_QKV,
    ROLE_ATTENTION_O,
    ROLE_MLP_GATE,
    ROLE_MLP_UP,
    ROLE_MLP_DOWN,
    ROLE_MOE_ROUTER,
    ROLE_MOE_EXPERT_GATE,
    ROLE_MOE_EXPERT_UP,
    ROLE_MOE_EXPERT_DOWN,
    ROLE_MOE_SHARED_GATE,
    ROLE_MOE_SHARED_UP,
    ROLE_MOE_SHARED_DOWN,
    ROLE_SSM_IN,
    ROLE_SSM_DT,
    ROLE_SSM_CONV,
    ROLE_SSM_OUT,
    ROLE_RNN_MIXER,
    ROLE_EMBEDDING,
    ROLE_NORM,
    ROLE_LM_HEAD,
    ROLE_OTHER,
)

DEFAULT_LORA_ROLES = (
    ROLE_ATTENTION_Q,
    ROLE_ATTENTION_K,
    ROLE_ATTENTION_V,
    ROLE_ATTENTION_QKV,
    ROLE_ATTENTION_O,
    ROLE_MLP_GATE,
    ROLE_MLP_UP,
    ROLE_MLP_DOWN,
    ROLE_MOE_EXPERT_GATE,
    ROLE_MOE_EXPERT_UP,
    ROLE_MOE_EXPERT_DOWN,
    ROLE_MOE_SHARED_GATE,
    ROLE_MOE_SHARED_UP,
    ROLE_MOE_SHARED_DOWN,
    ROLE_SSM_IN,
    ROLE_SSM_DT,
    ROLE_SSM_OUT,
    ROLE_RNN_MIXER,
)

LAYER_BANDS = ("all", "early", "middle", "late")
EXPERT_POLICIES = ("all", "dense", "routed", "shared", "router")
_ATTENTION_LEAF_ROLES = {
    "q_proj": ROLE_ATTENTION_Q,
    "query": ROLE_ATTENTION_Q,
    "wq": ROLE_ATTENTION_Q,
    "k_proj": ROLE_ATTENTION_K,
    "key": ROLE_ATTENTION_K,
    "wk": ROLE_ATTENTION_K,
    "v_proj": ROLE_ATTENTION_V,
    "value": ROLE_ATTENTION_V,
    "wv": ROLE_ATTENTION_V,
    "query_key_value": ROLE_ATTENTION_QKV,
    "qkv_proj": ROLE_ATTENTION_QKV,
    "c_attn": ROLE_ATTENTION_QKV,
    "o_proj": ROLE_ATTENTION_O,
    "out_proj": ROLE_ATTENTION_O,
    "wo": ROLE_ATTENTION_O,
}
_MLP_LEAF_ROLES = {
    "gate_proj": ROLE_MLP_GATE,
    "w1": ROLE_MLP_GATE,
    "up_proj": ROLE_MLP_UP,
    "w3": ROLE_MLP_UP,
    "fc1": ROLE_MLP_UP,
    "dense_h_to_4h": ROLE_MLP_UP,
    "c_fc": ROLE_MLP_UP,
    "down_proj": ROLE_MLP_DOWN,
    "w2": ROLE_MLP_DOWN,
    "fc2": ROLE_MLP_DOWN,
    "dense_4h_to_h": ROLE_MLP_DOWN,
}


@dataclass(frozen=True)
class SemanticTarget:
    """One parameter tensor that an update program may legally address.

    The role vocabulary is intentionally semantic rather than model-name based.
    It follows the low-rank update motivation of LoRA (Hu et al., 2021), the
    router/expert split in sparse MoE models (Fedus et al., 2021; Jiang et al.,
    2024; Dai et al., 2024), the state-mixing targets in SSM/RNN hybrids
    (Gu and Dao, 2023; Peng et al., 2023), and CODA's idea that optimization
    should operate on a constrained program space that can be lowered to the
    active backend (Guo et al., 2026).
    """

    name: str
    module_name: str
    parameter_name: str
    role: str
    shape: tuple[int, ...]
    module_class: str
    layer_index: int | None = None
    expert_index: int | None = None

    @property
    def parameter_count(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class ArchitectureProfile:
    model_class: str
    targets: tuple[SemanticTarget, ...]

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(role for role in CANONICAL_ROLES if any(target.role == role for target in self.targets))

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted({int(target.layer_index) for target in self.targets if target.layer_index is not None}))

    def role_counts(self) -> dict[str, int]:
        counts = {role: 0 for role in CANONICAL_ROLES}
        for target in self.targets:
            counts[target.role] = counts.get(target.role, 0) + 1
        return counts


@dataclass(frozen=True)
class LLMUpdateProgram:
    """Architecture-neutral BO candidate over LLM update targets."""

    roles: tuple[str, ...] = DEFAULT_LORA_ROLES
    layer_band: str = "all"
    expert_policy: str = "all"
    rank: int = 1
    scale: float = 1.0
    seed: int = 0
    max_targets: int | None = None

    def __post_init__(self) -> None:
        invalid_roles = tuple(role for role in self.roles if role not in CANONICAL_ROLES)
        if invalid_roles:
            raise ValueError(f"Unknown LLM update role(s): {invalid_roles}.")
        if self.layer_band not in LAYER_BANDS:
            raise ValueError(f"Unknown layer_band {self.layer_band!r}; expected one of {LAYER_BANDS}.")
        if self.expert_policy not in EXPERT_POLICIES:
            raise ValueError(f"Unknown expert_policy {self.expert_policy!r}; expected one of {EXPERT_POLICIES}.")
        if int(self.rank) < 1:
            raise ValueError("rank must be >= 1.")
        if float(self.scale) < 0.0:
            raise ValueError("scale must be nonnegative.")
        if self.max_targets is not None and int(self.max_targets) < 1:
            raise ValueError("max_targets must be None or >= 1.")


def discover_architecture_profile(model: Any, *, trainable_only: bool = False, include_bias: bool = False) -> ArchitectureProfile:
    targets: list[SemanticTarget] = []
    for module_name, module in model.named_modules():
        module_class = type(module).__name__
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if trainable_only and not bool(getattr(parameter, "requires_grad", False)):
                continue
            if not include_bias and str(parameter_name).endswith("bias"):
                continue
            shape = tuple(int(dim) for dim in getattr(parameter, "shape", ()))
            if not shape:
                continue
            full_name = f"{module_name}.{parameter_name}" if module_name else str(parameter_name)
            role = semantic_role(module_name=module_name, parameter_name=str(parameter_name), module_class=module_class)
            targets.append(
                SemanticTarget(
                    name=full_name,
                    module_name=str(module_name),
                    parameter_name=str(parameter_name),
                    role=role,
                    shape=shape,
                    module_class=module_class,
                    layer_index=_parse_layer_index(str(module_name)),
                    expert_index=_parse_expert_index(str(module_name)),
                )
            )
    return ArchitectureProfile(model_class=type(model).__name__, targets=tuple(targets))


def semantic_role(*, module_name: str, parameter_name: str = "weight", module_class: str = "") -> str:
    name = _clean_name(module_name)
    leaf = name.rsplit(".", 1)[-1] if name else ""
    param = str(parameter_name).lower()
    cls = str(module_class).lower()

    role = _structural_role(name, leaf, cls)
    if role is None and _is_ssm_path(name):
        role = _ssm_role(name, leaf, param, cls)
    if role is None:
        role = _projection_role(name, leaf)
    return role or ROLE_OTHER


def _structural_role(name: str, leaf: str, module_class: str) -> str | None:
    if "embed" in leaf or "embedding" in module_class:
        return ROLE_EMBEDDING
    if leaf in {"lm_head", "output_head"}:
        return ROLE_LM_HEAD
    if "norm" in leaf or "norm" in module_class:
        return ROLE_NORM
    if "router" in leaf or leaf in {"gate", "moe_gate"} or "router" in module_class:
        return ROLE_MOE_ROUTER
    if "shared_expert" in name:
        return _shared_expert_role(leaf)
    if ".experts." in name or ".expert." in name:
        return _routed_expert_role(leaf)
    return None


def _projection_role(name: str, leaf: str) -> str | None:
    if not _is_ssm_path(name) and leaf in _ATTENTION_LEAF_ROLES:
        return _ATTENTION_LEAF_ROLES[leaf]
    if "attention" in name and leaf in {"dense", "c_proj"}:
        return ROLE_ATTENTION_O
    if "mlp" in name and leaf in {"c_proj"}:
        return ROLE_MLP_DOWN
    return _MLP_LEAF_ROLES.get(leaf)


def resolve_update_program(profile: ArchitectureProfile, program: LLMUpdateProgram) -> tuple[SemanticTarget, ...]:
    targets = [
        target
        for target in profile.targets
        if target.role in program.roles and _layer_band_matches(profile, target, program.layer_band) and _expert_policy_matches(target, program.expert_policy)
    ]
    if program.max_targets is not None:
        targets = targets[: int(program.max_targets)]
    if not targets:
        raise ValueError(
            "LLM update program matched no targets: "
            f"roles={program.roles}, layer_band={program.layer_band!r}, expert_policy={program.expert_policy!r}, available_roles={profile.roles}"
        )
    return tuple(targets)


def update_program_features(profile: ArchitectureProfile, program: LLMUpdateProgram) -> tuple[float, ...]:
    matched = resolve_update_program(profile, program)
    counts = {role: 0 for role in CANONICAL_ROLES}
    for target in matched:
        counts[target.role] += 1
    denom = float(len(matched))
    role_features = tuple(float(counts[role]) / denom for role in CANONICAL_ROLES)
    layers = [target.layer_index for target in matched if target.layer_index is not None]
    mean_layer = _normalised_layer_position(profile, float(sum(layers) / len(layers))) if layers else 0.0
    experts = [target.expert_index for target in matched if target.expert_index is not None]
    mean_expert = float(sum(experts) / len(experts)) / max(float(max(experts) + 1), 1.0) if experts else 0.0
    return role_features + (
        mean_layer,
        mean_expert,
        math.log1p(float(program.rank)),
        float(program.scale),
        _seed_feature(int(program.seed)),
        float(len(matched)),
    )


def lora_target_module_names(profile: ArchitectureProfile, roles: tuple[str, ...] = DEFAULT_LORA_ROLES) -> tuple[str, ...]:
    names: list[str] = []
    for target in profile.targets:
        if target.role in roles and target.parameter_name == "weight" and target.module_name not in names:
            names.append(target.module_name)
    return tuple(names)


def coerce_update_roles(value: Any | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_LORA_ROLES
    if isinstance(value, str):
        if value.strip() in {"", "default"}:
            return DEFAULT_LORA_ROLES
        roles = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        roles = tuple(str(part).strip() for part in value if str(part).strip())
    if not roles:
        raise ValueError("LLM update roles must not be empty.")
    invalid = tuple(role for role in roles if role not in CANONICAL_ROLES)
    if invalid:
        raise ValueError(f"Unknown LLM update role(s): {invalid}.")
    return roles


def make_update_program(
    *,
    roles: Any | None = None,
    layer_band: str = "all",
    expert_policy: str = "all",
    rank: int = 1,
    scale: float = 1.0,
    seed: int = 0,
    max_targets: int | None = None,
) -> LLMUpdateProgram:
    return LLMUpdateProgram(
        roles=coerce_update_roles(roles),
        layer_band=str(layer_band),
        expert_policy=str(expert_policy),
        rank=int(rank),
        scale=float(scale),
        seed=int(seed),
        max_targets=max_targets,
    )


def _shared_expert_role(leaf: str) -> str:
    if leaf in {"gate_proj", "w1"}:
        return ROLE_MOE_SHARED_GATE
    if leaf in {"up_proj", "w3", "fc1"}:
        return ROLE_MOE_SHARED_UP
    if leaf in {"down_proj", "w2", "fc2"}:
        return ROLE_MOE_SHARED_DOWN
    return ROLE_OTHER


def _routed_expert_role(leaf: str) -> str:
    if leaf in {"gate_proj", "w1"}:
        return ROLE_MOE_EXPERT_GATE
    if leaf in {"up_proj", "w3", "fc1"}:
        return ROLE_MOE_EXPERT_UP
    if leaf in {"down_proj", "w2", "fc2"}:
        return ROLE_MOE_EXPERT_DOWN
    return ROLE_OTHER


def _ssm_role(name: str, leaf: str, parameter_name: str, module_class: str) -> str:
    if "time_mix" in name or "channel_mix" in name or "rwkv" in name:
        return ROLE_RNN_MIXER
    if "conv" in leaf or "conv" in module_class:
        return ROLE_SSM_CONV
    if leaf in {"dt_proj", "dt"} or "dt_" in leaf or parameter_name in {"dt", "dt_bias"}:
        return ROLE_SSM_DT
    if leaf in {"out_proj", "o_proj"}:
        return ROLE_SSM_OUT
    if leaf in {"in_proj", "x_proj", "z_proj"}:
        return ROLE_SSM_IN
    if "mixer" in name or "ssm" in name or "mamba" in name:
        return ROLE_RNN_MIXER
    return ROLE_OTHER


def _is_ssm_path(name: str) -> bool:
    lowered = str(name).lower()
    return any(part in lowered for part in ("mamba", "ssm", "mixer", "time_mix", "channel_mix"))


def _layer_band_matches(profile: ArchitectureProfile, target: SemanticTarget, band: str) -> bool:
    if band == "all" or target.layer_index is None:
        return band == "all"
    pos = _normalised_layer_position(profile, float(target.layer_index))
    if band == "early":
        return pos < 1.0 / 3.0
    if band == "middle":
        return 1.0 / 3.0 <= pos < 2.0 / 3.0
    if band == "late":
        return pos >= 2.0 / 3.0
    return False


def _expert_policy_matches(target: SemanticTarget, policy: str) -> bool:
    if policy == "all":
        return True
    if policy == "dense":
        return target.expert_index is None and not target.role.startswith("moe_")
    if policy == "routed":
        return target.expert_index is not None and target.role.startswith("moe_expert_")
    if policy == "shared":
        return target.role.startswith("moe_shared_")
    if policy == "router":
        return target.role == ROLE_MOE_ROUTER
    return False


def _normalised_layer_position(profile: ArchitectureProfile, layer_index: float) -> float:
    layers = profile.layers
    if not layers:
        return 0.0
    lo = float(layers[0])
    hi = float(layers[-1])
    if hi <= lo:
        return 0.0
    return (float(layer_index) - lo) / (hi - lo + 1.0)


def _parse_layer_index(module_name: str) -> int | None:
    match = re.search(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)(?:\.|$)", module_name)
    return None if match is None else int(match.group(1))


def _parse_expert_index(module_name: str) -> int | None:
    match = re.search(r"(?:^|\.)(?:experts|expert)\.(\d+)(?:\.|$)", module_name)
    return None if match is None else int(match.group(1))


def _clean_name(module_name: str) -> str:
    return str(module_name).lower().replace(".base_layer", "").replace(".linear", "")


def _seed_feature(seed: int) -> float:
    digest = hashlib.sha1(str(int(seed)).encode("ascii")).digest()
    return int.from_bytes(digest[:4], "big") / float(2**32 - 1)


__all__ = [
    "ArchitectureProfile",
    "CANONICAL_ROLES",
    "DEFAULT_LORA_ROLES",
    "LLMUpdateProgram",
    "SemanticTarget",
    "coerce_update_roles",
    "discover_architecture_profile",
    "lora_target_module_names",
    "make_update_program",
    "resolve_update_program",
    "semantic_role",
    "update_program_features",
]
