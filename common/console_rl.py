from __future__ import annotations

from typing import Any

from common.console_core import (
    _ALGO_SCHEMAS,
    _algo_metric_strings,
    _bold,
    _collect_header_config_items,
    _cyan,
    _dim,
    _format_sps,
    _global_step,
    _green,
    _print_config_table,
    _print_line,
)


def print_run_header(
    algo_name: str,
    config: Any,
    env: Any,
    training: Any,
    runtime: Any,
    *,
    step_label: str = "step",
    total_label: str = "iters",
    prefix: str = "",
) -> None:
    """Print a compact run header. algo_name selects metric columns (ppo, sac)."""
    obs_contract = getattr(getattr(env, "io_contract", None), "observation", None)
    if obs_contract is not None:
        from_pixels = obs_contract.mode == "pixels"
        if from_pixels:
            backbone = "nature_cnn_atari" if int(obs_contract.model_channels or 3) == 4 else "nature_cnn"
        else:
            backbone = getattr(config, "backbone_name", "mlp")
    else:
        from_pixels = bool(getattr(env.env_conf, "from_pixels", False))
        backbone = "nature_cnn" if from_pixels else getattr(config, "backbone_name", "mlp")
    algo_metrics = _ALGO_SCHEMAS.get(algo_name, [])

    _print_line(_dim("─" * 80), prefix=prefix)
    if algo_name == "ppo":
        title = (
            _bold(_cyan("PPO"))
            + f"  {config.env_tag}  seed={config.seed}  {runtime.device.type}  "
            + f"{training.frames_per_batch} frames/batch  {training.num_iterations} {total_label}"
        )
        _print_line(title, prefix=prefix)
        first_col = "iter"
    else:
        total = getattr(config, "total_timesteps", 0)
        title = _bold(_cyan(algo_name.upper())) + f"  {config.env_tag}  seed={config.seed}  {runtime.device.type}  " + f"total={total:,}"
        _print_line(title, prefix=prefix)
        first_col = "steps"
    _print_line(
        f"  obs={env.obs_dim} act={env.act_dim} backbone={backbone} from_pixels={from_pixels}",
        prefix=prefix,
    )
    cfg_items = _collect_header_config_items(config, training, runtime)
    cfg_cols = 3 if len(cfg_items) >= 18 else 2
    _print_config_table(cfg_items, prefix=prefix, cols=cfg_cols)
    _print_line(_dim("─" * 80), prefix=prefix)

    if algo_name == "ppo":
        parts = [
            f"{first_col:>5}",
            f"{step_label:>9}",
            f"{'eval':>7}",
            f"{'heldout':>7}",
            f"{'best':>7}",
        ]
    else:
        parts = [f"{first_col:>9}", f"{'eval':>7}", f"{'heldout':>7}", f"{'best':>7}"]
    for name, width, _ in algo_metrics:
        parts.append(name.rjust(width))
    parts.extend([f"{'time':>7}", f"{'sps':>6}"])
    _print_line(" ".join(parts), prefix=prefix)
    _print_line(_dim("-" * 80), prefix=prefix)


def print_iteration_log(
    iteration: int,
    num_iterations: int,
    frames_per_batch: int,
    *,
    eval_return: float | None = None,
    heldout_return: float | None = None,
    best_return: float = 0.0,
    algo_metrics: dict[str, float] | None = None,
    algo_name: str = "ppo",
    elapsed: float = 0.0,
    step_override: int | None = None,
    prefix: str = "",
) -> None:
    """Print a single iteration log line with eval and algo-specific metrics."""
    dash = "  -  "
    global_step = _global_step(iteration, frames_per_batch, step_override)
    eval_str = (f"{eval_return:.1f}" if eval_return is not None else "  -  ").rjust(7)
    heldout_str = (f"{heldout_return:.1f}" if heldout_return is not None else "  -  ").rjust(7)
    best_str = f"{best_return:.1f}".rjust(7)
    sps_str = _format_sps(global_step, elapsed, dash=dash)
    algo_strs = _algo_metric_strings(algo_name=algo_name, algo_metrics=algo_metrics, dash=dash)

    if step_override is not None:
        parts = [f"{global_step:9,d}", eval_str, heldout_str, best_str]
    else:
        parts = [
            f"{iteration:5d}",
            f"{global_step:9,d}",
            eval_str,
            heldout_str,
            best_str,
        ]
    parts.extend(algo_strs)
    parts.extend([f"{elapsed:6.1f}s".rjust(7), sps_str])
    _print_line(" ".join(parts), prefix=prefix)


def print_iteration_simple(
    iteration: int,
    num_iterations: int,
    frames_per_batch: int,
    elapsed: float,
    *,
    algo_name: str = "ppo",
    step_override: int | None = None,
    prefix: str = "",
) -> None:
    """Print a progress-only line (no eval this iteration)."""
    global_step = _global_step(iteration, frames_per_batch, step_override)
    dash = "  -  "
    sps_str = _format_sps(global_step, elapsed, dash=dash)
    algo_strs = _algo_metric_strings(algo_name=algo_name, algo_metrics=None, dash=dash)

    if step_override is not None:
        parts = [f"{global_step:9,d}", dash.rjust(7), dash.rjust(7), dash.rjust(7)]
    else:
        parts = [
            f"{iteration:5d}",
            f"{global_step:9,d}",
            dash.rjust(7),
            dash.rjust(7),
            dash.rjust(7),
        ]
    parts.extend(algo_strs)
    parts.extend([f"{elapsed:6.1f}s".rjust(7), sps_str])
    _print_line(" ".join(parts), prefix=prefix)


def print_run_footer(
    best_return: float,
    total_iters_or_steps: int,
    total_time: float,
    *,
    algo_name: str = "ppo",
    step_label: str = "iters",
) -> None:
    """Print run completion summary."""
    print(flush=True)
    print(_dim("─" * 80), flush=True)
    print(
        _green(f"Done  best_return={best_return:.2f}  {total_iters_or_steps} {step_label}  {total_time:.1f}s"),
        flush=True,
    )
    print(_dim("─" * 80), flush=True)
