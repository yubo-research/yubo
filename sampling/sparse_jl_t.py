"""Sparse JL transforms (torch). Implementation split across sparse_jl_t_*.py."""

from .sparse_jl_t_hash import CHUNK_SIZE as _CHUNK_SIZE
from .sparse_jl_t_hash import MASK64 as _MASK64
from .sparse_jl_t_transforms import (
    block_sparse_hash_scatter_from_nz_t as _block_sparse_hash_scatter_from_nz_t,
)
from .sparse_jl_t_transforms import (
    block_sparse_jl_noise_from_seed,
    block_sparse_jl_noise_from_seed_wr,
    block_sparse_jl_transform_module,
    block_sparse_jl_transform_module_to_cpu_wr,
    block_sparse_jl_transform_module_wr,
    block_sparse_jl_transform_t,
)

__all__ = [
    "_CHUNK_SIZE",
    "_MASK64",
    "_block_sparse_hash_scatter_from_nz_t",
    "block_sparse_jl_noise_from_seed",
    "block_sparse_jl_noise_from_seed_wr",
    "block_sparse_jl_transform_module",
    "block_sparse_jl_transform_module_to_cpu_wr",
    "block_sparse_jl_transform_module_wr",
    "block_sparse_jl_transform_t",
]
