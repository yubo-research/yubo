import numpy as np


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
        acc = inc ^ (int(seed) & mask) ^ ((int(j) & mask) << 1)
        chosen_rows = []
        while len(chosen_rows) < s_local:
            acc = (acc + inc) & mask
            z = acc
            z ^= z >> 30
            z = (z * m1) & mask
            z ^= z >> 27
            z = (z * m2) & mask
            z ^= z >> 31
            row = int(z % d_local)
            # linear membership check for tiny s
            duplicate = False
            for rr in chosen_rows:
                if rr == row:
                    duplicate = True
                    break
            if duplicate:
                continue
            chosen_rows.append(row)
            acc = (acc + inc) & mask
            z2 = acc
            z2 ^= z2 >> 30
            z2 = (z2 * m1) & mask
            z2 ^= z2 >> 27
            z2 = (z2 * m2) & mask
            z2 ^= z2 >> 31
            sign = 1.0 if (z2 & 1) == 1 else -1.0
            y[row] += sign * v * inv_sqrt_s
    return y
