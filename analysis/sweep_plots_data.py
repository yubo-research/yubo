"""Load and normalize sweep trace matrices from results directories."""

import json
import re
import sys
from pathlib import Path
from typing import Iterator

import numpy as np


def _iter_matching_run_dirs(
    exp_path: Path,
    regex_pattern: str,
    env_tag: str | None,
) -> Iterator[tuple[Path, int]]:
    rx = re.compile(regex_pattern)
    for subdir in sorted(exp_path.iterdir()):
        if not subdir.is_dir():
            continue
        config_file = subdir / "config.json"
        if not config_file.exists():
            continue
        with open(config_file) as f:
            config = json.load(f)
        if env_tag is not None and str(config.get("env_tag", "")) != str(env_tag):
            continue
        opt_name = config.get("opt_name", "")
        match = rx.search(opt_name)
        if not match:
            continue
        yield subdir, int(match.group(1))


def _load_traces_or_skip(
    subdir: Path,
    val: int,
    trace_key: str,
    param_name_for_print: str,
):
    sp = sys.modules.get("analysis.sweep_plots")
    if sp is not None:
        load = sp.load_traces
    else:
        from analysis.data_sets import load_traces as load

    try:
        traces = load(str(subdir), key=trace_key)
    except Exception as e:
        print(f"Skipping {param_name_for_print}={val}: {e}")
        return None
    if traces is None or len(traces) == 0:
        print(f"No traces for {param_name_for_print}={val}")
        return None
    trace_matrix = np.asarray(traces)
    if trace_matrix.ndim == 1:
        if trace_matrix.size <= 1 or np.all(np.isnan(trace_matrix)):
            print(f"Skipping {param_name_for_print}={val}: incomplete trace shape {trace_matrix.shape}")
            return None
        trace_matrix = trace_matrix[np.newaxis, :]
    if trace_matrix.ndim != 2 or trace_matrix.shape[1] == 0:
        print(f"Skipping {param_name_for_print}={val}: unsupported trace shape {trace_matrix.shape}")
        return None
    return trace_matrix


def _collect_plot_curves_data(
    exp_path: Path,
    regex_pattern: str,
    env_tag: str | None,
    trace_key: str,
    param_name_for_print: str,
) -> tuple[list[int], list] | None:
    param_values: list[int] = []
    all_curves: list = []
    for subdir, val in _iter_matching_run_dirs(exp_path, regex_pattern, env_tag):
        traces = _load_traces_or_skip(subdir, val, trace_key, param_name_for_print)
        if traces is None:
            continue
        y_best_curves = np.maximum.accumulate(traces, axis=1)
        param_values.append(val)
        all_curves.append(y_best_curves)
    if not param_values:
        return None
    sort_idx = np.argsort(param_values)
    param_values = [param_values[i] for i in sort_idx]
    all_curves = [all_curves[i] for i in sort_idx]
    return param_values, all_curves
