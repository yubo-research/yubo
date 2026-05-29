from __future__ import annotations

from typing import Any

import numpy as np


def be_sim_batch_enabled(objective: Any) -> bool:
    return bool(getattr(objective, "_vectorize", False)) and hasattr(objective, "evaluate_many")


def be_pick_candidate(
    objective: Any,
    candidates: np.ndarray,
    *,
    seed: int,
) -> tuple[int, float, float] | None:
    if not be_sim_batch_enabled(objective):
        return None
    means, ses = objective.evaluate_many(np.asarray(candidates, dtype=np.float64), seed=int(seed))
    best = int(np.argmax(means))
    return best, float(means[best]), float(ses[best])


def be_pick_mezo_seed(
    objective: Any,
    x_plus: np.ndarray,
    x_minus: np.ndarray,
    seeds: list[int],
    *,
    sigma: float,
) -> int | None:
    """Pick MeZO BE seed via batched sim when the objective supports evaluate_many."""
    if not be_sim_batch_enabled(objective):
        return None
    n = len(seeds)
    if n == 0:
        return None
    plus = np.asarray(x_plus, dtype=np.float64)
    minus = np.asarray(x_minus, dtype=np.float64)
    base = int(seeds[0])
    mu_plus, se_plus = objective.evaluate_many(plus, seed=base)
    mu_minus, se_minus = objective.evaluate_many(minus, seed=base + n)
    two_sigma = 2.0 * float(sigma)
    grad = (mu_plus - mu_minus) / two_sigma
    se_grad = np.sqrt(se_plus**2 + se_minus**2) / two_sigma
    return int(np.argmax(np.abs(grad) + se_grad))
