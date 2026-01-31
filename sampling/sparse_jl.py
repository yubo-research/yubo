import numpy as np


def _mix64(z: int, *, mask: int, m1: int, m2: int) -> int:
    z &= mask
    z ^= z >> 30
    z = (z * m1) & mask
    z ^= z >> 27
    z = (z * m2) & mask
    z ^= z >> 31
    return z


def _choose_rows_and_signs(*, seed: int, j: int, d_local: int, s_local: int, inc: int, m1: int, m2: int, mask: int):
    acc = inc ^ (int(seed) & mask) ^ ((int(j) & mask) << 1)
    chosen = []
    chosen_rows = []
    while len(chosen_rows) < s_local:
        acc = (acc + inc) & mask
        z = _mix64(acc, mask=mask, m1=m1, m2=m2)
        row = int(z % d_local)
        duplicate = False
        for rr in chosen_rows:
            if rr == row:
                duplicate = True
                break
        if duplicate:
            continue
        chosen_rows.append(row)
        acc = (acc + inc) & mask
        z2 = _mix64(acc, mask=mask, m1=m1, m2=m2)
        sign = 1.0 if (z2 & 1) == 1 else -1.0
        chosen.append((row, sign))
    return chosen


def block_sparse_jl_transform(x: np.ndarray, d: int, s: int = 4, seed: int = 42) -> np.ndarray:
    assert x.ndim == 1
    assert d > 0
    assert s > 0
    if s > d:
        raise ValueError("s must be <= d")
    D = x.shape[0]
    if not np.any(x):
        return np.zeros(d, dtype=float)
    inv_sqrt_s = 1.0 / np.sqrt(s)
    y = np.zeros(d, dtype=float)
    inc = 0x9E3779B97F4A7C15
    m1 = 0xBF58476D1CE4E5B9
    m2 = 0x94D049BB133111EB
    mask = (1 << 64) - 1
    d_local = int(d)
    s_local = int(s)
    for j in range(D):
        v = x[j]
        if v == 0.0:
            continue
        for row, sign in _choose_rows_and_signs(
            seed=seed,
            j=j,
            d_local=d_local,
            s_local=s_local,
            inc=inc,
            m1=m1,
            m2=m2,
            mask=mask,
        ):
            y[row] += sign * v * inv_sqrt_s
    return y
