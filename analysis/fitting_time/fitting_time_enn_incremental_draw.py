"""Streaming synthetic train/test draws matching :func:`draw_benchmark_synthetic_xy`."""

from __future__ import annotations

import numpy as np
import torch

from .evaluate_draw import SYNTHETIC_BENCHMARK_N_TEST, _batch_pure_env_reward
from .evaluate_metrics import (
    SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME,
    env_action_coords_to_surrogate_unit_x,
    normalize_benchmark_function_name,
)


def _train_xy_unit_cube_segment(
    *,
    D: int,
    function_name: str,
    problem_seed: int,
    n_train: int,
    start_row: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rows ``[start_row, n_train)`` of the training draw for ``N = n_train``."""
    if start_row < 0 or start_row >= n_train:
        raise ValueError(f"invalid segment start_row={start_row} for n_train={n_train}")
    fn = normalize_benchmark_function_name(function_name)
    d = int(D)
    base = int(problem_seed)
    end = int(n_train)
    start = int(start_row)
    torch.manual_seed(base)
    x = torch.rand(end, d) * 2.0 - 1.0
    x_seg = x[start:end]
    if fn == SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME:
        x_u = env_action_coords_to_surrogate_unit_x(x)
        y_full = torch.sin(2 * torch.pi * x_u).mean(dim=1, keepdim=True) + 0.1 * torch.randn(end, 1)
    else:
        from problems import pure_functions

        env_tag = f"f:{fn}-{d}d"
        env = pure_functions.make(env_tag, problem_seed=base, distort=True)
        y_body = _batch_pure_env_reward(env, x.detach().cpu().numpy().astype(np.float64))
        y_full = torch.tensor(y_body, dtype=torch.float64) + 0.1 * torch.randn(end, 1)
        x_u = env_action_coords_to_surrogate_unit_x(x)
    x_out = env_action_coords_to_surrogate_unit_x(x_seg).detach().cpu().numpy().astype(np.float64)
    y_out = y_full[start:end].detach().cpu().numpy().astype(np.float64)
    return x_out, y_out


def draw_benchmark_test_xy_unit_cube(
    *,
    D: int,
    function_name: str,
    problem_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    fn = normalize_benchmark_function_name(function_name)
    base = int(problem_seed)
    n_test = int(SYNTHETIC_BENCHMARK_N_TEST)
    if fn == SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME:
        torch.manual_seed(base + 1)
        x_test = torch.rand(n_test, D) * 2.0 - 1.0
        x_test_u = env_action_coords_to_surrogate_unit_x(x_test)
        y_test = torch.sin(2 * torch.pi * x_test_u).mean(dim=1, keepdim=True) + 0.1 * torch.randn(n_test, 1)
        return (
            x_test_u.detach().cpu().numpy().astype(np.float64),
            y_test.detach().cpu().numpy().astype(np.float64),
        )
    from problems import pure_functions

    env_tag = f"f:{fn}-{D}d"
    env = pure_functions.make(env_tag, problem_seed=base, distort=True)
    torch.manual_seed(base + 1)
    x_test = torch.rand(n_test, D) * 2.0 - 1.0
    y_test_body = _batch_pure_env_reward(env, x_test.detach().cpu().numpy().astype(np.float64))
    y_test = torch.tensor(y_test_body, dtype=torch.float64) + 0.1 * torch.randn(n_test, 1)
    x_test_u = env_action_coords_to_surrogate_unit_x(x_test)
    return (
        x_test_u.detach().cpu().numpy().astype(np.float64),
        y_test.detach().cpu().numpy().astype(np.float64),
    )
