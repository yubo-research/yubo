from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FitnessSummary:
    fitnesses: np.ndarray
    normalized: np.ndarray
    per_prompt_mean: np.ndarray
    per_population_mean: np.ndarray
    mean: float
    min: float
    max: float
    normalized_std: float


def validate_eggroll_population(
    *,
    population_size: int,
    num_engines: int,
    samples_per_prompt: int,
    temperature: float,
    pass_at_k: bool,
) -> None:
    if int(population_size) < 2:
        raise ValueError("population_size must be >= 2.")
    if int(population_size) % 2 != 0:
        raise ValueError("population_size must be even for antithetic EggRoll sampling.")
    if int(num_engines) < 1:
        raise ValueError("num_engines must be >= 1.")
    if int(population_size) % int(num_engines) != 0:
        raise ValueError(f"population_size={population_size} must be divisible by num_engines={num_engines}.")
    loras_per_engine = int(population_size) // int(num_engines)
    if int(loras_per_engine) % 2 != 0:
        raise ValueError(
            "population_size/num_engines must be even so each engine gets antithetic (+/-) pairs. "
            f"Got population_size={population_size}, num_engines={num_engines} -> {loras_per_engine} arms/engine."
        )
    if int(samples_per_prompt) > 1 and float(temperature) <= 0.0:
        raise ValueError("samples_per_prompt > 1 requires temperature > 0.")
    if bool(pass_at_k) and int(samples_per_prompt) <= 1:
        raise ValueError("pass_at_k=true requires samples_per_prompt > 1.")


def num_iterations_from_budget(
    *,
    num_rounds: int | None,
    total_timesteps: int | None,
    population_size: int,
    prompt_batch_size: int,
) -> int:
    if num_rounds is not None:
        return int(num_rounds)
    if total_timesteps is None:
        raise ValueError("EggRoll requires num_rounds or total_timesteps.")
    steps_per_iter = max(1, int(population_size) * int(prompt_batch_size))
    return max(1, int(np.ceil(int(total_timesteps) / steps_per_iter)))


def summarize_fitness(fitnesses: Any, *, normalize_with_std: bool) -> FitnessSummary:
    arr = np.asarray(fitnesses, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"fitnesses must have shape (population_size, prompt_batch_size), got {arr.shape}.")
    per_prompt_mean = np.mean(arr, axis=0, keepdims=True)
    per_population_mean = np.mean(arr, axis=1)
    normalized = np.mean(arr - per_prompt_mean, axis=1)
    normalized_std = float(np.std(normalized))
    if normalize_with_std:
        normalized = normalized / (normalized_std + 1e-8)
    return FitnessSummary(
        fitnesses=arr,
        normalized=normalized,
        per_prompt_mean=per_prompt_mean,
        per_population_mean=per_population_mean,
        mean=float(np.mean(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        normalized_std=normalized_std,
    )


__all__ = [
    "FitnessSummary",
    "num_iterations_from_budget",
    "summarize_fitness",
    "validate_eggroll_population",
]
