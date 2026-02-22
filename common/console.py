"""Unified console output for RL and BO runs. Algorithm/optimizer-specific metrics supported."""

from __future__ import annotations

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


def _fmt_metric(val: float | None, fmt: str, width: int) -> str:
    if val is None or (val != val):  # nan
        return "  -  ".rjust(width)
    return ("{:" + fmt + "}").format(val).rjust(width)


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
        from_pixels = obs_contract.mode == "pixels"
        if from_pixels:
            backbone = "nature_cnn_atari" if int(obs_contract.model_channels or 3) == 4 else "nature_cnn"
        else:
            backbone = getattr(config, "backbone_name", "mlp")
    else:
        from_pixels = bool(getattr(env.env_conf, "from_pixels", False))
        backbone = "nature_cnn" if from_pixels else getattr(config, "backbone_name", "mlp")
    algo_metrics = _ALGO_SCHEMAS.get(algo_name, [])

    def _prefixed(s: str) -> str:
        return f"{prefix}{s}" if prefix else s

    print(_prefixed(_dim("─" * 80)), flush=True)
    if algo_name == "ppo":
        title = (
            _bold("PPO")
            + f"  {config.env_tag}  seed={config.seed}  {runtime.device.type}  "
            + f"{training.frames_per_batch} frames/batch  {training.num_iterations} {total_label}",
        )
        print(_prefixed(title), flush=True)
        first_col = "iter"
    else:
        total = getattr(config, "total_timesteps", 0)
        title = (_bold(algo_name.upper()) + f"  {config.env_tag}  seed={config.seed}  {runtime.device.type}  " + f"total={total:,}",)
        print(_prefixed(title), flush=True)
        first_col = "step"
    print(_prefixed(_dim(f"  obs={env.obs_dim} act={env.act_dim} backbone={backbone} from_pixels={from_pixels}")), flush=True)
    print(_prefixed(_dim("─" * 80)), flush=True)

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
    print(_prefixed(" ".join(parts)), flush=True)
    print(_prefixed(_dim("-" * 80)), flush=True)


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
    if step_override is not None:
        global_step = step_override
    else:
        global_step = iteration * frames_per_batch
    sps = float(global_step / elapsed) if elapsed > 0 else float("nan")
    eval_str = (f"{eval_return:.1f}" if eval_return is not None else "  -  ").rjust(7)
    heldout_str = (f"{heldout_return:.1f}" if heldout_return is not None else "  -  ").rjust(7)
    best_str = f"{best_return:.1f}".rjust(7)
    sps_str = (f"{sps:.0f}" if not (sps != sps) else "  -  ").rjust(6)

    schema = _ALGO_SCHEMAS.get(algo_name, [])
    algo_strs = []
    if algo_metrics:
        for name, width, fmt in schema:
            algo_strs.append(_fmt_metric(algo_metrics.get(name), fmt, width))
    else:
        algo_strs = [_fmt_metric(None, "", w) for _, w, _ in schema]

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
    line = " ".join(parts)
    print(f"{prefix}{line}" if prefix else line, flush=True)


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
    if step_override is not None:
        global_step = step_override
    else:
        global_step = iteration * frames_per_batch
    sps = float(global_step / elapsed) if elapsed > 0 else float("nan")
    dash = "  -  "
    sps_str = (f"{sps:.0f}" if not (sps != sps) else dash).rjust(6)

    schema = _ALGO_SCHEMAS.get(algo_name, [])
    algo_strs = [dash.rjust(w) for _, w, _ in schema]

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
    line = " ".join(parts)
    print(f"{prefix}{line}" if prefix else line, flush=True)


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


def _schema_for(opt_name: str) -> list[tuple[str, int, str]]:
    """Resolve opt_name to metric schema. Supports prefix matching."""
    if opt_name in _OPT_SCHEMAS:
        return _OPT_SCHEMAS[opt_name]
    if opt_name.startswith("turbo-enn-multi"):
        return MULTI_TURBO_METRICS
    if "turbo-enn" in opt_name or "turbo-enn-fit" in opt_name or "turbo-enn-p" in opt_name:
        return TURBO_METRICS
    return []


def _parse_iter_line(line: str) -> dict[str, Any] | None:
    """Parse ITER line; return dict of fields or None if not an ITER line."""
    s = line.strip()
    if not s.startswith("ITER:"):
        return None
    out: dict[str, Any] = {}
    for part in s[5:].split():
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        v_clean = v.rstrip("s")
        try:
            out[k] = float(v_clean)
        except ValueError:
            out[k] = v_clean
    if "iter" not in out:
        return None
    return out


def print_bo_header_top(env_tag: str, opt_name: str, num_rounds: int, num_arms: int) -> None:
    """Print BO run header (above problem description)."""
    print(_dim("-" * 72), flush=True)
    print(
        _bold("BO") + f"  {env_tag}  {opt_name}  rounds={num_rounds}  arms={num_arms}",
        flush=True,
    )
    print(_dim("-" * 72), flush=True)


def print_bo_round(parsed: dict[str, Any], opt_name: str) -> None:
    """Print one compact log line per round. Uses ret_best, ret_eval labels."""
    schema = _schema_for(opt_name)
    try:
        ret_eval = float(parsed.get("ret_eval", 0))
        ret_best = float(parsed.get("ret_best", 0))
    except (TypeError, ValueError):
        ret_eval = ret_best = 0.0
    i = int(parsed.get("iter", 0))
    elapsed = parsed.get("elapsed", 0)
    proposal_dt = parsed.get("proposal_dt", parsed.get("dt_prop"))
    proposal_elapsed = parsed.get("proposal_elapsed")
    parts = [f"[{i}]", f"ret_best={ret_best:.1f}", f"ret_eval={ret_eval:.1f}"]
    for key, _, fmt in schema:
        val = parsed.get(key)
        if isinstance(val, (int, float)) and val == val:
            parts.append(f"{key}={val:{fmt}}")
    if isinstance(proposal_dt, (int, float)) and proposal_dt == proposal_dt:
        parts.append(f"proposal_dt={float(proposal_dt):.3f}s")
    if isinstance(proposal_elapsed, (int, float)) and proposal_elapsed == proposal_elapsed:
        parts.append(f"proposal_elapsed={float(proposal_elapsed):.3f}s")
    parts.append(f"elapsed={float(elapsed):.1f}s")
    print("  ".join(parts), flush=True)


def print_bo_footer(best_return: float, total_time: float) -> None:
    """Print run completion summary."""
    print(flush=True)
    print(_dim("-" * 72), flush=True)
    print(_green(f"Done  ret_best={best_return:.1f}  time={total_time:.1f}s"), flush=True)
    print(_dim("-" * 72), flush=True)


class BOConsoleCollector:
    """Collector that formats ITER lines as compact log lines. Implements Collector interface."""

    def __init__(
        self,
        env_tag: str,
        opt_name: str,
        num_rounds: int,
        num_arms: int,
        *,
        inner: Any = None,
    ):
        self._lines = deque()
        self._env_tag = env_tag
        self._opt_name = opt_name
        self._num_rounds = num_rounds
        self._num_arms = num_arms
        self._inner = inner
        self._header_printed = False
        self._pre_header_buffer: list[str] = []
        self._best_return = -1e99

    def _flush_pre_header_and_print_header(self) -> None:
        """Print buffered lines (PROBLEM, algo output) then BO header."""
        for buf_line in self._pre_header_buffer:
            print(buf_line, flush=True)
        self._pre_header_buffer.clear()
        if not self._header_printed:
            print_bo_header_top(self._env_tag, self._opt_name, self._num_rounds, self._num_arms)
            self._header_printed = True

    def __call__(self, line: str) -> None:
        self._lines.append(line)
        parsed = _parse_iter_line(line)
        if parsed is not None:
            self._flush_pre_header_and_print_header()
            try:
                r = float(parsed.get("ret_best", parsed.get("ret_eval", -1e99)))
                if r > self._best_return:
                    self._best_return = r
            except (TypeError, ValueError):
                pass
            print_bo_round(parsed, self._opt_name)
        else:
            if self._header_printed:
                print(line, flush=True)
            else:
                self._pre_header_buffer.append(line)
        if self._inner is not None:
            self._inner(line)

    def __iter__(self):
        return iter(self._lines)
