from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParameterMetadata:
    name: str
    shape: tuple[int, ...]
    kind: str  # "MM", "EMB", "CONV", "NORM", "BIAS", "GATE", "OTHER"


@dataclass(frozen=True)
class UniversalSubspaceTemplate:
    parameters: list[ParameterMetadata]
    search_dim: int
    seed: int
    basis_leaf: list[int]
    basis_index: list[int]
    basis_sign: list[float]


def discover_vllm_parameters(model: Any) -> list[ParameterMetadata]:
    """Principled architecture discovery by traversing the module tree."""
    from torch import nn

    norm_types: list[type[nn.Module]] = [nn.LayerNorm, nn.GroupNorm]
    rms_norm = getattr(nn, "RMSNorm", None)
    if rms_norm is not None:
        norm_types.append(rms_norm)

    type_map = {
        (nn.Linear,): "MM",
        (nn.Embedding,): "EMB",
        tuple(norm_types): "NORM",
        (nn.Conv1d, nn.Conv2d, nn.Conv3d): "CONV",
    }

    metadata = []
    seen_params = set()

    for module_name, module in model.named_modules():
        kind = "OTHER"
        for types, candidate_kind in type_map.items():
            if isinstance(module, types):
                kind = candidate_kind
                break

        if "Gate" in module.__class__.__name__:
            kind = "GATE"

        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            if full_name in seen_params:
                continue
            param_kind = "BIAS" if "bias" in param_name.lower() else kind
            metadata.append(
                ParameterMetadata(
                    name=full_name,
                    shape=tuple(int(dim) for dim in param.shape),
                    kind=param_kind,
                )
            )
            seen_params.add(full_name)

    for name, param in model.named_parameters():
        if name not in seen_params and param.requires_grad:
            metadata.append(
                ParameterMetadata(
                    name=name,
                    shape=tuple(int(dim) for dim in param.shape),
                    kind="OTHER",
                )
            )

    return metadata


def build_universal_subspace_template(
    *,
    parameters: list[ParameterMetadata],
    search_dim: int,
    seed: int,
    lora_only: bool = False,
    basis_max_leaves: int | None = None,
) -> UniversalSubspaceTemplate:
    """Builds a coordinate-based subspace template over all model parameters."""
    import numpy as np

    indices = []
    sizes = []
    for i, parameter in enumerate(parameters):
        size = int(np.prod(parameter.shape))
        if size <= 0:
            continue
        if lora_only and parameter.kind not in ("MM", "EMB"):
            continue
        indices.append(i)
        sizes.append(size)

    indices = np.array(indices)
    sizes = np.array(sizes)

    if indices.size == 0:
        raise ValueError("No eligible parameters found for universal subspace.")

    rng = np.random.default_rng(int(seed))
    probs = sizes.astype(np.float64) / sizes.sum()

    basis_leaf_idx = rng.choice(indices.size, size=int(search_dim), replace=True, p=probs)
    basis_leaf = indices[basis_leaf_idx].tolist()
    basis_index = [int(rng.integers(sizes[idx])) for idx in basis_leaf_idx]
    basis_sign = rng.choice([-1.0, 1.0], size=int(search_dim)).astype(float).tolist()

    return UniversalSubspaceTemplate(
        parameters=parameters,
        search_dim=int(search_dim),
        seed=int(seed),
        basis_leaf=basis_leaf,
        basis_index=basis_index,
        basis_sign=basis_sign,
    )


def universal_subspace_seed(
    *,
    base_seed: int,
    num_pop_pairs: int,
    search_dim: int,
    pop_step: int,
    pop_pair_idx: int,
) -> int:
    """Seed schedule shared by universal-subspace perturb/apply paths.

    Keep this in lock-step across:
    - evaluation-time perturbations (per arm)
    - ES update reconstruction (per population pair)
    """

    return int(base_seed) + int(num_pop_pairs) * int(search_dim) * int(pop_step) + int(pop_pair_idx) * int(search_dim)


def universal_global_pop_pair_idx(
    *,
    engine_rank: int,
    num_engines: int,
    population_size: int,
    local_pop_pair_idx: int,
) -> int:
    """Maps an engine-local pop-pair index to the global pop-pair index."""

    if int(num_engines) < 1:
        raise ValueError("num_engines must be >= 1.")
    if int(population_size) % int(num_engines) != 0:
        raise ValueError("population_size must be divisible by num_engines.")
    loras_per_engine = int(population_size) // int(num_engines)
    if int(loras_per_engine) % 2 != 0:
        raise ValueError("population_size/num_engines must be even for antithetic sampling.")
    num_pairs_per_engine = int(loras_per_engine) // 2
    return int(engine_rank) * int(num_pairs_per_engine) + int(local_pop_pair_idx)


__all__ = [
    "ParameterMetadata",
    "UniversalSubspaceTemplate",
    "build_universal_subspace_template",
    "discover_vllm_parameters",
    "universal_global_pop_pair_idx",
    "universal_subspace_seed",
]
