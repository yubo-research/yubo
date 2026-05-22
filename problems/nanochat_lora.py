from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _NanochatSubspaceCodec:
    """Sparse low-dimensional coordinates over a nanochat GPT model state."""

    def __init__(
        self,
        model: nn.Module,
        *,
        dim: int,
        delta_scale: float,
        seed: int,
    ) -> None:
        self.model = model
        self.dim = int(dim)
        self.delta_scale = float(delta_scale)

        # Identify target parameters: weight matrices in Linear layers
        # Standard nanochat names: transformer.h.i.attn.c_q.weight, etc.
        target_params = []
        target_kinds = []
        for name, param in model.named_parameters():
            if "weight" in name and isinstance(param, (nn.Parameter, torch.Tensor)):
                # Filter for Linear layers and Embedding
                # Kind 1: Matrix Multiplication (MM) - eligible for low-rank eggroll noise
                # Kind 0: Standard/Embedding - eligible for flat noise
                if any(x in name for x in ["c_q", "c_k", "c_v", "c_proj", "c_fc", "lm_head"]):
                    target_params.append((name, param))
                    target_kinds.append(1)
                elif "wte" in name:
                    target_params.append((name, param))
                    target_kinds.append(0)

        if not target_params:
            raise ValueError("nanochat model has no target parameters for UHD search.")

        # Capture actual current sizes (after any padding/initialization)
        sizes = np.asarray([int(p.numel()) for _, p in target_params], dtype=np.int64)
        probs = sizes.astype(np.float64) / sizes.sum()

        rng = np.random.default_rng(int(seed))
        self._names = tuple(name for name, _ in target_params)
        self._leaf_shapes = tuple(tuple(int(v) for v in p.shape) for _, p in target_params)
        self._leaf_kind = np.asarray(target_kinds, dtype=np.int64)

        # Pick which tensor each subspace dimension affects
        self._basis_tensor_idx = rng.choice(len(target_params), size=self.dim, replace=True, p=probs).astype(np.int64)

        # Pick which specific weight in that tensor
        # USE THE ACTUAL RECORDED SIZE FOR THE CHOSEN TENSOR
        self._basis_flat_idx = np.asarray(
            [rng.integers(sizes[idx]) for idx in self._basis_tensor_idx],
            dtype=np.int64,
        )

        # Pick random signs
        self._basis_sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=self.dim).astype(np.float32)

        self.num_total_params = int(sizes.sum())
        self._base_noise_seed = int(seed)
        self.x0 = np.zeros((self.dim,), dtype=np.float64)

    @torch.no_grad()
    def apply(self, x: np.ndarray) -> None:
        """Apply the low-dimensional perturbation x to the model in-place."""
        coeffs = np.asarray(x, dtype=np.float32).reshape(-1)
        if coeffs.shape[0] != self.dim:
            raise ValueError(f"x must have shape ({self.dim},), got {coeffs.shape}.")

        active = np.flatnonzero(coeffs != 0.0)
        if active.size == 0:
            return

        # Group updates by tensor to minimize memory operations
        param_dict = dict(self.model.named_parameters())
        unique_tensors, tensor_starts = np.unique(self._basis_tensor_idx[active], return_index=True)

        for tensor_idx in unique_tensors:
            name = self._names[int(tensor_idx)]
            param = param_dict[name]

            # Find all coordinates in x affecting this specific tensor
            mask = self._basis_tensor_idx[active] == tensor_idx
            positions = active[mask]

            # Map indices to torch
            flat_indices = torch.as_tensor(self._basis_flat_idx[positions], dtype=torch.long, device=param.device)
            values = torch.as_tensor(
                coeffs[positions] * self._basis_sign[positions] * self.delta_scale,
                dtype=param.dtype,
                device=param.device,
            )

            # Apply update to the flattened parameter
            param.view(-1).index_add_(0, flat_indices, values)

    @torch.no_grad()
    def revert(self, x: np.ndarray) -> None:
        """Revert the perturbation x from the model in-place."""
        self.apply(-x)

    def sample_eggroll_direction(
        self,
        *,
        seed: int,
        rank: int,
        freeze_nonlora: bool = False,
    ) -> np.ndarray:
        """Generates a structured eggroll-style direction for the subspace."""
        rank = max(int(rank), 1)
        direction = np.zeros(self.dim, dtype=np.float64)

        # Use NumPy for high-speed targeted sampling
        # (This replaces the slow JAX fold_in/vmap approach)
        rng = np.random.default_rng(int(seed))

        for leaf_idx in np.unique(self._basis_tensor_idx):
            positions = np.flatnonzero(self._basis_tensor_idx == int(leaf_idx))
            kind = int(self._leaf_kind[int(leaf_idx)])

            if freeze_nonlora and kind == 0:
                continue

            shape = self._leaf_shapes[int(leaf_idx)]

            if kind == 1 and len(shape) >= 2:
                # Targeted low-rank generation using NumPy
                direction[positions] = _numpy_targeted_low_rank_values(
                    rng,
                    flat_indices=self._basis_flat_idx[positions],
                    shape=shape,
                    rank=rank,
                )
            else:
                # Targeted flat generation using NumPy
                direction[positions] = rng.standard_normal(len(positions))

        return direction.astype(np.float64)


def _numpy_targeted_low_rank_values(rng: np.random.Generator, *, flat_indices: np.ndarray, shape: tuple[int, ...], rank: int) -> np.ndarray:
    """Efficiently generate low-rank values using NumPy."""
    cols = int(shape[-1])
    rows_count = int(np.prod(shape[:-1]))

    row_ids = np.asarray(flat_indices, dtype=np.int64) // cols
    col_ids = np.asarray(flat_indices, dtype=np.int64) % cols

    # We still want to be "SOTA" by only generating what's needed

    # Sample row and column factors
    a_full = rng.standard_normal((rows_count, rank)).astype(np.float32)
    b_full = rng.standard_normal((cols, rank)).astype(np.float32)

    # Targeted dot product
    a = a_full[row_ids]
    b = b_full[col_ids]

    values = np.sum(a * b, axis=1) / np.sqrt(float(rank))
    return values.astype(np.float64)
