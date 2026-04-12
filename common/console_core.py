from __future__ import annotations

import dataclasses
import shutil
import sys
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
    excluded = {
        "env_tag",
        "seed",
        "backbone_name",
        "total_timesteps",
        "from_pixels",
        "pixels_only",
        "theta_dim",
        "exp_dir",
        "device",
    }
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
        if key in excluded or not _is_displayable_config_value(value):
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


PPO_METRICS = [("kl", 7, ".4f"), ("clipfrac", 8, ".4f")]
SAC_METRICS = [("actor", 7, ".4f"), ("critic", 7, ".4f"), ("alpha", 7, ".4f")]

_ALGO_SCHEMAS: dict[str, list[tuple[str, int, str]]] = {
    "ppo": PPO_METRICS,
    "sac": SAC_METRICS,
}


def register_algo_metrics(algo_name: str, metrics: list[tuple[str, int, str]]) -> None:
    """Register algo-specific metric columns. metrics: [(name, width, fmt), ...]"""
    _ALGO_SCHEMAS[algo_name] = metrics


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
