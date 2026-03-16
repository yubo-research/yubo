from __future__ import annotations

from typing import Any


def update_best_actor_if_improved(
    *, eval_return: float, best_return: float, best_actor_state: dict[str, Any] | None, capture_actor_state
) -> tuple[float, dict[str, Any] | None, bool]:
    eval_return_f = float(eval_return)
    best_return_f = float(best_return)
    if eval_return_f > best_return_f:
        return (eval_return_f, capture_actor_state(), True)
    return (best_return_f, best_actor_state, False)


def evaluate_heldout_with_best_actor(
    *,
    best_actor_state: dict[str, Any] | None,
    num_denoise_passive: int | None,
    heldout_i_noise: int,
    with_actor_state,
    evaluate_for_best,
    eval_env_conf,
    eval_policy,
) -> float | None:
    if num_denoise_passive is None or best_actor_state is None:
        return None
    with with_actor_state(best_actor_state):
        return float(evaluate_for_best(eval_env_conf, eval_policy, int(num_denoise_passive), i_noise=int(heldout_i_noise)))
