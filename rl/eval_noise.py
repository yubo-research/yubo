from __future__ import annotations

from dataclasses import dataclass

FROZEN_HELDOUT_NOISE_INDEX = 99999


@dataclass(frozen=True)
class EvalPlan:
    eval_seed: int
    heldout_i_noise: int


def normalize_eval_noise_mode(eval_noise_mode: str | None) -> str:
    normalized = str(eval_noise_mode if eval_noise_mode is not None else "frozen").strip().lower()
    if normalized not in {"frozen", "natural"}:
        raise ValueError("eval_noise_mode must be one of: frozen, natural")
    return normalized


def eval_index_for_due_step(*, current: int, interval: int) -> int:
    current_i = int(current)
    interval_i = int(interval)
    if current_i <= 0:
        raise ValueError(f"current must be > 0, got {current_i}.")
    if interval_i <= 0:
        raise ValueError(f"interval must be > 0, got {interval_i}.")
    if current_i % interval_i != 0:
        raise ValueError(f"current={current_i} is not divisible by interval={interval_i}.")
    return int(current_i // interval_i)


def resolve_eval_seed(
    *,
    seed: int,
    eval_seed_base: int | None,
    eval_noise_mode: str | None,
    eval_index: int,
) -> int:
    eval_index_i = int(eval_index)
    if eval_index_i <= 0:
        raise ValueError(f"eval_index must be > 0, got {eval_index_i}.")
    base_seed = int(eval_seed_base if eval_seed_base is not None else seed)
    mode = normalize_eval_noise_mode(eval_noise_mode)
    if mode == "frozen":
        return base_seed
    return int(base_seed + eval_index_i - 1)


def resolve_heldout_noise_index(*, eval_noise_mode: str | None, eval_seed: int) -> int:
    mode = normalize_eval_noise_mode(eval_noise_mode)
    if mode == "frozen":
        return int(FROZEN_HELDOUT_NOISE_INDEX)
    return int(eval_seed)


def build_eval_plan(
    *,
    current: int,
    interval: int,
    seed: int,
    eval_seed_base: int | None,
    eval_noise_mode: str | None,
) -> EvalPlan:
    eval_index = eval_index_for_due_step(current=current, interval=interval)
    eval_seed = resolve_eval_seed(
        seed=seed,
        eval_seed_base=eval_seed_base,
        eval_noise_mode=eval_noise_mode,
        eval_index=eval_index,
    )
    heldout_i_noise = resolve_heldout_noise_index(
        eval_noise_mode=eval_noise_mode,
        eval_seed=eval_seed,
    )
    return EvalPlan(
        eval_seed=int(eval_seed),
        heldout_i_noise=int(heldout_i_noise),
    )
