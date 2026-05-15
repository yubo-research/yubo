import torch

# Splitmix64 constants (signed int64 for PyTorch compatibility)
_GOLDEN = 0x9E3779B97F4A7C15 - (1 << 64)
_MIX_M1 = 0xBF58476D1CE4E5B9 - (1 << 64)
_MIX_M2 = 0x94D049BB133111EB - (1 << 64)

CHUNK_SIZE = 1 << 20  # 1M elements per sub-chunk
MASK64 = (1 << 64) - 1


def wrap_int64(val: int) -> int:
    """Wrap an arbitrary-precision Python int to signed int64 range."""
    val = val & MASK64
    return val - (1 << 64) if val >= (1 << 63) else val


def vmix64_inplace(z: torch.Tensor, tmp: torch.Tensor) -> None:
    """In-place splitmix64 finalizer. Modifies z, uses tmp as scratch."""
    torch.bitwise_right_shift(z, 30, out=tmp)
    tmp.bitwise_and_((1 << 34) - 1)
    z.bitwise_xor_(tmp)
    z.mul_(_MIX_M1)
    torch.bitwise_right_shift(z, 27, out=tmp)
    tmp.bitwise_and_((1 << 37) - 1)
    z.bitwise_xor_(tmp)
    z.mul_(_MIX_M2)
    torch.bitwise_right_shift(z, 31, out=tmp)
    tmp.bitwise_and_((1 << 33) - 1)
    z.bitwise_xor_(tmp)


def vmix64(z: torch.Tensor) -> torch.Tensor:
    """Vectorized splitmix64 output finalizer (allocating version)."""
    z = z ^ ((z >> 30) & ((1 << 34) - 1))
    z = z * _MIX_M1
    z = z ^ ((z >> 27) & ((1 << 37) - 1))
    z = z * _MIX_M2
    z = z ^ ((z >> 31) & ((1 << 33) - 1))
    return z


def compute_rows_and_signs(
    global_indices: torch.Tensor,
    d: int,
    s: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute target rows (without replacement) and signs for coordinates.

    Uses Fisher-Yates-style selection: slot k draws from d-k candidates
    and remaps past previously chosen rows.

    Returns rows (N, s) int64 and signs (N, s) float32.
    """
    n = global_indices.shape[0]
    rows = torch.empty(n, s, dtype=torch.int64, device=device)
    signs = torch.empty(n, s, dtype=torch.float32, device=device)

    h = torch.empty(n, dtype=torch.int64, device=device)
    tmp = torch.empty(n, dtype=torch.int64, device=device)
    cand = torch.empty(n, dtype=torch.int64, device=device)

    for k in range(s):
        offset = wrap_int64(seed * _GOLDEN + k * _GOLDEN)
        torch.add(global_indices, offset, out=h)
        vmix64_inplace(h, tmp)

        torch.remainder(h, d - k, out=cand)

        torch.bitwise_right_shift(h, 32, out=tmp)
        tmp.bitwise_and_(1)
        signs[:, k] = torch.where(tmp == 1, 1.0, -1.0)

        if k > 0:
            prev_sorted, _ = torch.sort(rows[:, :k], dim=1)
            for p in range(k):
                cand.add_((cand >= prev_sorted[:, p]).long())

        rows[:, k] = cand

    return rows, signs


def compute_rows_and_signs_wr(
    global_indices: torch.Tensor,
    d: int,
    s: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute target rows (with replacement) and signs for coordinates."""
    n = global_indices.shape[0]
    rows = torch.empty(n, s, dtype=torch.int64, device=device)
    signs = torch.empty(n, s, dtype=torch.float32, device=device)

    h = torch.empty(n, dtype=torch.int64, device=device)
    tmp = torch.empty(n, dtype=torch.int64, device=device)

    for k in range(s):
        offset = wrap_int64(seed * _GOLDEN + k * _GOLDEN)
        torch.add(global_indices, offset, out=h)
        vmix64_inplace(h, tmp)
        torch.remainder(h, d, out=rows[:, k])

        torch.bitwise_right_shift(h, 32, out=tmp)
        tmp.bitwise_and_(1)
        signs[:, k] = torch.where(tmp == 1, 1.0, -1.0)

    return rows, signs
