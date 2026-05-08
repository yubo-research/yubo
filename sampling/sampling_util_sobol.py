import numpy as np
from scipy.stats import qmc


def _sobol_random_n(num_dim: int, n: int, *, scramble: bool = True, seed=None) -> np.ndarray:
    """
    Draw n Sobol points in [0, 1]^d without SciPy's "n must be a power of 2" warning.

    SciPy's `Sobol.random(n)` emits a warning when n is not a power of 2. We instead draw
    `2**m` points via `random_base2(m)` (which expects powers of 2) and slice.
    """
    if n <= 0:
        return np.empty((0, num_dim), dtype=np.float64)

    engine = qmc.Sobol(num_dim, scramble=scramble, seed=seed)
    m = (n - 1).bit_length()  # smallest m with 2**m >= n (m=0 when n=1)
    return engine.random_base2(m)[:n]
