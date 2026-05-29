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
    eval_label: str = "eval",
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
        collector = getattr(config, "collector", None)
        total = getattr(collector, "total_frames", getattr(config, "total_timesteps", 0))
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
            f"{eval_label:>7}",
            f"{'heldout':>7}",
            f"{'best':>7}",
        ]
    else:
        parts = [
            f"{first_col:>9}",
            f"{eval_label:>7}",
            f"{'heldout':>7}",
            f"{'best':>7}",
        ]
    for name, width, _ in algo_metrics:
        parts.append(name.rjust(width))
    parts.extend([f"{'train':>6}", f"{'eval':>6}", f"{'time':>7}", f"{'sps':>6}"])
    _print_line(" ".join(parts), prefix=prefix)
    _print_line(_dim("-" * 80), prefix=prefix)


def _format_phase_dt(value: float | None, *, width: int = 6) -> str:
    dash = "   -  "
    if value is None or (isinstance(value, float) and value != value):
        return dash.rjust(width)
    return f"{float(value):.2f}s".rjust(width)


def _resolve_global_step(
    iteration: int,
    frames_per_batch: int,
    *,
    step_override: int | None,
    global_step: int | None,
) -> int:
    if step_override is not None:
        return int(step_override)
    if global_step is not None:
        return int(global_step)
    return _global_step(iteration, frames_per_batch, None)


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
    train_dt: float | None = None,
    eval_dt: float | None = None,
    global_step: int | None = None,
    step_override: int | None = None,
    prefix: str = "",
) -> None:
    """Print a single iteration log line with eval and algo-specific metrics."""
    dash = "  -  "
    resolved_step = _resolve_global_step(
        iteration,
        frames_per_batch,
        step_override=step_override,
        global_step=global_step,
    )
    eval_str = (f"{eval_return:.1f}" if eval_return is not None else "  -  ").rjust(7)
    heldout_str = (f"{heldout_return:.1f}" if heldout_return is not None else "  -  ").rjust(7)
    best_str = f"{best_return:.1f}".rjust(7)
    sps_str = _format_sps(resolved_step, elapsed, dash=dash)
    algo_strs = _algo_metric_strings(algo_name=algo_name, algo_metrics=algo_metrics, dash=dash)

    if step_override is not None:
        parts = [f"{resolved_step:9,d}", eval_str, heldout_str, best_str]
    else:
        parts = [
            f"{iteration:5d}",
            f"{resolved_step:9,d}",
            eval_str,
            heldout_str,
            best_str,
        ]
    parts.extend(algo_strs)
    parts.extend(
        [
            _format_phase_dt(train_dt),
            _format_phase_dt(eval_dt),
            f"{elapsed:6.1f}s".rjust(7),
            sps_str,
        ]
    )
    _print_line(" ".join(parts), prefix=prefix)


def print_iteration_simple(
    iteration: int,
    num_iterations: int,
    frames_per_batch: int,
    elapsed: float,
    *,
    eval_return: float | None = None,
    best_return: float | None = None,
    algo_metrics: dict[str, float] | None = None,
    algo_name: str = "ppo",
    train_dt: float | None = None,
    eval_dt: float | None = None,
    global_step: int | None = None,
    step_override: int | None = None,
    prefix: str = "",
) -> None:
    """Print a progress-only line (no eval this iteration)."""
    resolved_step = _resolve_global_step(
        iteration,
        frames_per_batch,
        step_override=step_override,
        global_step=global_step,
    )
    dash = "  -  "
    eval_str = (f"{eval_return:.1f}" if eval_return is not None else dash).rjust(7)
    best_str = (f"{best_return:.1f}" if best_return is not None else dash).rjust(7)
    sps_str = _format_sps(resolved_step, elapsed, dash=dash)
    algo_strs = _algo_metric_strings(algo_name=algo_name, algo_metrics=algo_metrics, dash=dash)

    if step_override is not None:
        parts = [f"{resolved_step:9,d}", eval_str, dash.rjust(7), best_str]
    else:
        parts = [
            f"{iteration:5d}",
            f"{resolved_step:9,d}",
            eval_str,
            dash.rjust(7),
            best_str,
        ]
    parts.extend(algo_strs)
    parts.extend(
        [
            _format_phase_dt(train_dt),
            _format_phase_dt(eval_dt),
            f"{elapsed:6.1f}s".rjust(7),
            sps_str,
        ]
    )
    _print_line(" ".join(parts), prefix=prefix)


def print_iter_record(
    record: dict[str, float | int],
    *,
    algo_name: str,
    prefix: str = "",
) -> None:
    """Print one RL metrics record as an aligned table row (console standard)."""
    iteration = int(record.get("iter", 0))
    step = int(record.get("step", 0))
    iter_n = max(1, iteration)
    frames_per_batch = int(record.get("frames_per_iter", step // iter_n if step else 1))
    elapsed = float(record.get("elapsed", 0.0))
    train_dt = record.get("iter_dt")
    eval_dt = record.get("eval_dt")
    train_f = float(train_dt) if train_dt is not None else None
    eval_f = float(eval_dt) if eval_dt is not None else None
    algo_metrics = {name: float(record[name]) for name, _, _ in _ALGO_SCHEMAS.get(algo_name, []) if name in record}
    has_eval = "ret_eval" in record
    use_step_only = algo_name != "ppo"
    print_iteration_log(
        iteration,
        0,
        frames_per_batch,
        eval_return=float(record["ret_eval"]) if has_eval else None,
        heldout_return=float(record["ret_heldout"]) if "ret_heldout" in record else None,
        best_return=float(record.get("ret_best", 0.0)),
        algo_metrics=algo_metrics or None,
        algo_name=algo_name,
        elapsed=elapsed,
        train_dt=train_f,
        eval_dt=eval_f,
        global_step=step,
        step_override=step if use_step_only else None,
        prefix=prefix,
    )


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
