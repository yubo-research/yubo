from __future__ import annotations

from typing import Any

from rl.eval_noise import plan


def update_best(
    *,
    eval_return: float,
    best_return: float,
    best_actor_state: dict[str, Any] | None,
    capture_actor_state,
) -> tuple[float, dict[str, Any] | None, bool]:
    eval_return_f = float(eval_return)
    best_return_f = float(best_return)
    if eval_return_f > best_return_f:
        return (eval_return_f, capture_actor_state(), True)
    return (best_return_f, best_actor_state, False)


def heldout(
    *,
    best_actor_state: dict[str, Any] | None,
    num_denoise_passive: int | None,
    heldout_i_noise: int,
    with_actor_state,
    best,
    eval_env_conf,
    eval_policy,
) -> float | None:
    if num_denoise_passive is None or best_actor_state is None:
        return None
    with with_actor_state(best_actor_state):
        return float(
            best(
                eval_env_conf,
                eval_policy,
                int(num_denoise_passive),
                i_noise=int(heldout_i_noise),
            )
        )


def run(
    *,
    current: int,
    interval: int,
    seed: int,
    eval_seed_base: int | None,
    eval_noise_mode: str | None,
    state: Any,
    evaluate_actor,
    capture_actor_state,
    evaluate_heldout,
):
    ep = plan(
        current=int(current),
        interval=int(interval),
        seed=int(seed),
        eval_seed_base=eval_seed_base,
        eval_noise_mode=eval_noise_mode,
    )
    state.last_eval_return = evaluate_actor(eval_seed=int(ep.eval_seed))
    state.best_return, state.best_actor_state, _ = update_best(
        eval_return=float(state.last_eval_return),
        best_return=float(state.best_return),
        best_actor_state=state.best_actor_state,
        capture_actor_state=capture_actor_state,
    )
    state.last_heldout_return = evaluate_heldout(
        best_actor_state=state.best_actor_state,
        heldout_i_noise=int(ep.heldout_i_noise),
    )
    return ep


__all__ = ["heldout", "plan", "run", "update_best"]
