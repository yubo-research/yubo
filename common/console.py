from __future__ import annotations

import dataclasses
import re
import shutil
import sys
from collections import deque
from typing import Any

# ---- Shared helpers ----


def _is_tty() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _dim(s: str) -> str:
    if _is_tty():
        return f"\033[2m{s}\033[0m"
    return s


def _bold(s: str) -> str:
    if _is_tty():
        return f"\033[1m{s}\033[0m"
    return s


def _green(s: str) -> str:
    if _is_tty():
        return f"\033[32m{s}\033[0m"
    return s


def _cyan(s: str) -> str:
    if _is_tty():
        return f"\033[36m{s}\033[0m"
    return s


def _fmt_metric(val: float | None, fmt: str, width: int) -> str:
    if val is None or (val != val):  # nan
        return "  -  ".rjust(width)
    return ("{:" + fmt + "}").format(val).rjust(width)


def _with_prefix(line: str, prefix: str) -> str:
    return f"{prefix}{line}" if prefix else line


def _print_line(line: str, *, prefix: str = "") -> None:
    print(_with_prefix(line, prefix), flush=True)


def _cfg_value(value: Any) -> str:
    max_len = 20
    if value is None:
        return "-"
    if isinstance(value, float):
        if value != value:
            return "-"
        text = f"{value:g}"
    else:
        text = str(value)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


# Non-identity aliases only; keys not listed fall back to themselves.
_CFG_KEY_ALIASES = {
    "noise_seed_0": "noise_seed",
    "num_envs": "envs",
    "frames_per_batch": "fpb",
    "num_iterations": "iters",
    "batch_size": "batch",
    "learning_starts": "warmup",
    "update_every": "upd_every",
    "updates_per_step": "upd_per_step",
    "learning_rate": "lr",
    "learning_rate_actor": "lr_actor",
    "learning_rate_critic": "lr_critic",
    "learning_rate_alpha": "lr_alpha",
    "eval_interval": "eval_int",
    "eval_interval_steps": "eval_int",
    "log_interval": "log_int",
    "log_interval_steps": "log_int",
    "num_denoise": "denoise",
    "num_denoise_passive": "denoise_passive",
    "eval_seed_base": "eval_seed",
    "eval_noise_mode": "noise_mode",
    "collector_backend": "collector",
    "single_env_backend": "single_env",
    "collector_workers": "workers",
    "vector_backend": "vector",
    "vector_num_workers": "v_workers",
    "vector_batch_size": "v_batch",
    "replay_backend": "replay",
}


def _cfg_key(key: str) -> str:
    return _CFG_KEY_ALIASES.get(str(key), str(key))


def _print_config_table(
    items: list[tuple[str, Any]],
    *,
    prefix: str = "",
    cols: int = 2,
    key_width: int = 16,
) -> None:
    _ = key_width
    filtered = [(_cfg_key(str(k)), _cfg_value(v)) for k, v in items if v is not None]
    if not filtered:
        return
    cells = [f"{k}={v}" for k, v in filtered]
    term_width = int(shutil.get_terminal_size(fallback=(100, 20)).columns)
    available = max(52, min(110, term_width - len(prefix)))
    max_cols = max(1, min(3, int(cols), len(cells)))
    chosen_cols = 1
    cell_width = max((len(c) for c in cells), default=1)
    for cand_cols in range(max_cols, 0, -1):
        cand_width = max((len(cells[i]) for i in range(0, len(cells))), default=1)
        line_width = 2 + cand_cols * cand_width + 2 * (cand_cols - 1)
        if line_width <= available:
            chosen_cols = cand_cols
            cell_width = cand_width
            break
    _print_line(f"  {_cyan('config')}:", prefix=prefix)
    row_cells: list[list[str]] = []
    for i in range(0, len(cells), chosen_cols):
        row_cells.append(cells[i : i + chosen_cols])
    for cells in row_cells:
        padded = list(cells) + [""] * (chosen_cols - len(cells))
        line = "  " + "  ".join(f"{cell:<{cell_width}}" for cell in padded).rstrip()
        _print_line(line, prefix=prefix)


def _config_to_mapping(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    return {}


def _is_displayable_config_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (tuple, list)):
        return all(isinstance(v, (str, int, float, bool)) for v in value)
    return False


def _hide_header_key(key: str) -> bool:
    key = str(key)
    if key in {
        "env_tag",
        "seed",
        "backbone_name",
        "total_timesteps",
        "obs_mode",
        "theta_dim",
        "exp_dir",
        "device",
    }:
        return True
    return key.endswith("_hidden_sizes") or key.endswith("_activation") or key.endswith("_layer_norm") or key in {"critic_backbone_name", "share_backbone"}


def _collect_header_config_items(config: Any, training: Any, runtime: Any) -> list[tuple[str, Any]]:
    values = _config_to_mapping(config)
    if not bool(values.get("video_enable", False)):
        for key in list(values):
            if str(key).startswith("video_"):
                values.pop(key, None)
    if not bool(values.get("profile_enable", False)):
        for key in list(values):
            if str(key).startswith("profile_"):
                values.pop(key, None)
    if values.get("resume_from") in {None, "", False}:
        values.pop("resume_from", None)
    # Show resolved runtime settings when available (for torchrl runtime).
    for key in ("collector_backend", "single_env_backend", "collector_workers"):
        resolved = getattr(runtime, key, None)
        if resolved is not None and resolved != "n/a":
            values[key] = resolved
    # Ensure loop-shape fields are visible for basic headers.
    for key in ("frames_per_batch", "num_iterations"):
        raw = getattr(training, key, None)
        if raw is not None:
            values[key] = raw
    out: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for key, value in values.items():
        if _hide_header_key(key) or not _is_displayable_config_value(value):
            continue
        shown = _cfg_key(str(key))
        if shown in seen:
            shown = str(key)
        seen.add(shown)
        out.append((shown, value))
    return out


def _global_step(iteration: int, frames_per_batch: int, step_override: int | None) -> int:
    if step_override is not None:
        return int(step_override)
    return int(iteration * frames_per_batch)


def _format_sps(global_step: int, elapsed: float, *, width: int = 6, dash: str = "  -  ") -> str:
    sps = float(global_step / elapsed) if elapsed > 0 else float("nan")
    return (f"{sps:.0f}" if not (sps != sps) else dash).rjust(width)


def _algo_metric_strings(
    *,
    algo_name: str,
    algo_metrics: dict[str, float] | None,
    dash: str = "  -  ",
) -> list[str]:
    schema = _ALGO_SCHEMAS.get(algo_name, [])
    if algo_metrics:
        return [_fmt_metric(algo_metrics.get(name), fmt, width) for name, width, fmt in schema]
    return [dash.rjust(width) for _, width, _ in schema]


# ---- RL (algo) metrics ----

PPO_METRICS = [("kl", 7, ".4f"), ("clipfrac", 8, ".4f")]
SAC_METRICS = [("actor", 7, ".4f"), ("critic", 7, ".4f"), ("alpha", 7, ".4f")]

_ALGO_SCHEMAS: dict[str, list[tuple[str, int, str]]] = {
    "ppo": PPO_METRICS,
    "sac": SAC_METRICS,
}


def register_algo_metrics(algo_name: str, metrics: list[tuple[str, int, str]]) -> None:
    """Register algo-specific metric columns. metrics: [(name, width, fmt), ...]"""
    _ALGO_SCHEMAS[algo_name] = metrics


# ---- RL console ----


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
        obs_mode = str(obs_contract.mode)
        if obs_mode == "pixels":
            backbone = "nature_cnn_atari" if int(obs_contract.model_channels or 3) == 4 else "nature_cnn"
        else:
            backbone = getattr(config, "backbone_name", "mlp")
    else:
        obs_mode = str(getattr(env.env_conf, "obs_mode", "vector"))
        backbone = "nature_cnn" if obs_mode in {"image", "mixed", "pixels", "pixels+state"} else getattr(config, "backbone_name", "mlp")
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
        f"  obs={env.obs_dim} act={env.act_dim} backbone={backbone} obs_mode={obs_mode}",
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


# ---- BO (opt) metrics ----

TURBO_METRICS = [
    ("tr_length", 8, ".3f"),
    ("tr_obs", 6, ".0f"),
    ("fit_dt", 6, ".3f"),
    ("select_dt", 6, ".3f"),
]
MULTI_TURBO_METRICS = [
    ("tr_length", 8, ".3f"),
    ("fit_dt", 6, ".3f"),
    ("select_dt", 6, ".3f"),
    ("region_idx", 6, ".0f"),
    ("region_alloc", 8, ".0f"),
]
CMA_METRICS = [("sigma", 7, ".4f")]

_OPT_SCHEMAS: dict[str, list[tuple[str, int, str]]] = {
    "cma": CMA_METRICS,
    "turbo-enn-multi": MULTI_TURBO_METRICS,
}


def register_opt_metrics(opt_name: str, metrics: list[tuple[str, int, str]]) -> None:
    """Register opt-specific metric columns. metrics: [(key, width, fmt), ...]"""
    _OPT_SCHEMAS[opt_name] = metrics


def _parse_iter_line(line: str) -> dict[str, Any] | None:
    """Parse ITER line; return dict of fields or None if not an ITER line."""
    s = line.strip()
    if not s.startswith("ITER:"):
        return None
    out: dict[str, Any] = {}
    for m in re.finditer(r"(\w+)\s*=\s*(\S+)", s[5:]):
        k, v = m.group(1), m.group(2)
        v_clean = v.rstrip("s")
        try:
            out[k] = float(v_clean)
        except ValueError:
            out[k] = v_clean
    if "iter" not in out:
        return None
    return out


def print_bo_footer(best_return: float, total_time: float) -> None:
    """Print run completion summary."""
    print(flush=True)
    print(_dim("-" * 72), flush=True)
    print(_green(f"Done  ret_best={best_return:.1f}  time={total_time:.1f}s"), flush=True)
    print(_dim("-" * 72), flush=True)


class BOConsoleCollector:
    """Collector that echoes ITER lines to stdout (same format as data file). Implements Collector interface."""

    def __init__(self, *, inner: Any = None):
        self._lines = deque()
        self._inner = inner
        self._header_printed = False
        self._pre_header_buffer: list[str] = []
        self._best_return = -1e99

    def _flush_pre_header(self) -> None:
        """Print buffered lines (PROBLEM, algo output)."""
        for buf_line in self._pre_header_buffer:
            print(buf_line, flush=True)
        self._pre_header_buffer.clear()
        self._header_printed = True

    def __call__(self, line: str) -> None:
        self._lines.append(line)
        parsed = _parse_iter_line(line)
        if parsed is not None:
            self._flush_pre_header()
            try:
                r = float(parsed.get("ret_best", parsed.get("ret_eval", -1e99)))
                if r > self._best_return:
                    self._best_return = r
            except (TypeError, ValueError):
                pass
            print(line, flush=True)
        else:
            if self._header_printed:
                print(line, flush=True)
            else:
                self._pre_header_buffer.append(line)
        if self._inner is not None:
            self._inner(line)

    def __iter__(self):
        return iter(self._lines)
