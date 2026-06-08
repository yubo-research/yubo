from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class _LoraSubspaceCodec:
    """Sparse low-dimensional coordinates over a PEFT LoRA adapter state."""

    def __init__(
        self,
        template,
        *,
        dim: int,
        delta_scale: float,
        seed: int,
        basis_max_tensors: int | None,
    ) -> None:
        self.template = template
        self.dim = int(dim)
        self.delta_scale = float(delta_scale)

        leaves = [(name, value) for name, value in template.state_dict.items() if _is_search_tensor(name)]
        if not leaves:
            leaves = list(template.state_dict.items())
        if not leaves:
            raise ValueError("LoRA template has no trainable tensors for UHD text search.")

        sizes = np.asarray([int(tensor.numel()) for _, tensor in leaves], dtype=np.int64)
        valid = np.flatnonzero(sizes > 0)
        if valid.size == 0:
            raise ValueError("LoRA template tensors are all empty.")
        if basis_max_tensors is not None and int(basis_max_tensors) < valid.size:
            rng_for_tensors = np.random.default_rng(int(seed) ^ 0x9E3779B9)
            tensor_probs = sizes[valid].astype(np.float64)
            tensor_probs = tensor_probs / tensor_probs.sum()
            valid = np.sort(rng_for_tensors.choice(valid, size=int(basis_max_tensors), replace=False, p=tensor_probs).astype(np.int64))

        probs = sizes[valid].astype(np.float64)
        probs = probs / probs.sum()
        rng = np.random.default_rng(int(seed))
        self._names = tuple(name for name, _ in leaves)
        self._basis_tensor = rng.choice(valid, size=self.dim, replace=True, p=probs).astype(np.int64)
        self._basis_index = np.asarray(
            [rng.integers(sizes[tensor_idx]) for tensor_idx in self._basis_tensor],
            dtype=np.int64,
        )
        self._basis_sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=self.dim).astype(np.float32)
        self.num_total_tensors = int(len(leaves))
        self.num_candidate_tensors = int(valid.size)
        self.num_candidate_params = int(sizes[valid].sum())
        self.num_basis_tensors = int(np.unique(self._basis_tensor).size)
        self.x0 = np.zeros((self.dim,), dtype=np.float64)

    def decode(self, x: np.ndarray) -> dict[str, Any]:
        coeffs = np.asarray(x, dtype=np.float32).reshape(-1)
        if coeffs.shape[0] != self.dim:
            raise ValueError(f"x must have shape ({self.dim},), got {coeffs.shape}.")
        state = {name: tensor.clone() for name, tensor in self.template.state_dict.items()}
        active = np.flatnonzero(coeffs != 0.0)
        if active.size == 0:
            return state
        return self._decode_active(state, coeffs, active)

    def _decode_active(self, state: dict[str, Any], coeffs: np.ndarray, active: np.ndarray) -> dict[str, Any]:
        import torch

        for tensor_idx in np.unique(self._basis_tensor[active]):
            positions = active[self._basis_tensor[active] == tensor_idx]
            name = self._names[int(tensor_idx)]
            flat = state[name].reshape(-1)
            idx = torch.as_tensor(self._basis_index[positions], dtype=torch.long, device=flat.device)
            values = torch.as_tensor(
                coeffs[positions] * self._basis_sign[positions] * self.delta_scale,
                dtype=flat.dtype,
                device=flat.device,
            )
            flat.index_add_(0, idx, values)
        return state


def _is_search_tensor(name: str) -> bool:
    return ".lora_B." in str(name)


def _write_lora_adapter(adapter_path: Path, state: dict[str, Any], config: dict[str, Any]) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("Text UHD adapter materialization requires safetensors.") from exc

    with open(adapter_path / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f)
    save_file(
        {_adapter_tensor_name(name): tensor for name, tensor in state.items()},
        str(adapter_path / "adapter_model.safetensors"),
    )


def _adapter_tensor_name(name: str) -> str:
    return str(name).replace(".lora_A.default.weight", ".lora_A.weight").replace(".lora_B.default.weight", ".lora_B.weight")
