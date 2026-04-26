"""Synthetic train/test draws for surrogate benchmarks."""

from __future__ import annotations

import zlib

import numpy as np
import torch

from .evaluate_metrics import (
    SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME,
    env_action_coords_to_surrogate_unit_x,
    normalize_benchmark_function_name,
)


def _batch_pure_env_reward(env, actions_np: np.ndarray) -> np.ndarray:
    """Evaluate ``PureFunctionEnv`` rewards for each row of ``actions_np`` (N, D) in [-1, 1]."""
    n = actions_np.shape[0]
    y = np.empty((n, 1), dtype=np.float64)
    for i in range(n):
        step = env.step(actions_np[i])
        y[i, 0] = float(step.reward)
    return y


def synthetic_benchmark_data_seed(
    *,
    function_name: str,
    problem_seed: int,
    rep_index: int = 0,
) -> int:
    """Deterministic draw seed that varies by config seed, function, and replicate."""
    fn = normalize_benchmark_function_name(function_name)
    fn_offset = zlib.crc32(fn.encode("utf-8")) % 1_000_003
    seed = int(problem_seed) + 1_000_003 * int(rep_index) + fn_offset
    return seed % 2_147_483_647


def draw_benchmark_synthetic_xy(
    *,
    N: int,
    D: int,
    function_name: str,
    problem_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw train/test batches; ``x`` and ``x_test`` are in ``[-1, 1]`` (env / action scale)."""
    fn = normalize_benchmark_function_name(function_name)
    base = int(problem_seed)
    if fn == SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME:
        torch.manual_seed(base)
        x = torch.rand(N, D) * 2.0 - 1.0
        x_u = env_action_coords_to_surrogate_unit_x(x)
        y = torch.sin(2 * torch.pi * x_u).mean(dim=1, keepdim=True) + 0.1 * torch.randn(N, 1)
        torch.manual_seed(base + 1)
        x_test = torch.rand(N, D) * 2.0 - 1.0
        x_test_u = env_action_coords_to_surrogate_unit_x(x_test)
        y_test = torch.sin(2 * torch.pi * x_test_u).mean(dim=1, keepdim=True) + 0.1 * torch.randn(N, 1)
        return x, y, x_test, y_test
    from problems import pure_functions

    env_tag = f"f:{fn}-{D}d"
    env = pure_functions.make(env_tag, problem_seed=base, distort=True)
    torch.manual_seed(base)
    x = torch.rand(N, D) * 2.0 - 1.0
    y_body = _batch_pure_env_reward(env, x.detach().cpu().numpy().astype(np.float64))
    y = torch.tensor(y_body, dtype=torch.float64) + 0.1 * torch.randn(N, 1)
    torch.manual_seed(base + 1)
    x_test = torch.rand(N, D) * 2.0 - 1.0
    y_test_body = _batch_pure_env_reward(env, x_test.detach().cpu().numpy().astype(np.float64))
    y_test = torch.tensor(y_test_body, dtype=torch.float64) + 0.1 * torch.randn(N, 1)
    return x, y, x_test, y_test
