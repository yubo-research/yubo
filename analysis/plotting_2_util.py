"""Utility functions for plotting_2 module."""

import json
from pathlib import Path

import numpy as np

from analysis.plot_util import (
    collect_config_rows,
    normalize_results_and_exp_dir,
    uniq_int,
)


def noise_label(problem: str) -> str:
    if problem.endswith(":fn"):
        return "Frozen noise"
    return "Natural noise"


def speedup_x_label(cum_dt_prop_final_by_opt: dict[str, float] | None, problem: str) -> str | None:
    if not cum_dt_prop_final_by_opt:
        return None

    baseline_candidates = ("turbo-one", "turbo-one-na", "turbo-one-f")
    baseline_opt = next((o for o in baseline_candidates if o in cum_dt_prop_final_by_opt), None)
    if baseline_opt is None:
        return None

    compare_opt = "turbo-enn-p" if problem.endswith(":fn") else "turbo-enn-fit/acq_type=ucb"
    if compare_opt not in cum_dt_prop_final_by_opt:
        return None

    t_baseline = cum_dt_prop_final_by_opt.get(baseline_opt, None)
    t_compare = cum_dt_prop_final_by_opt.get(compare_opt, None)
    if t_baseline is None or t_compare is None or not np.isfinite(t_baseline) or not np.isfinite(t_compare) or t_compare <= 0:
        return None

    x = int(round(float(t_baseline) / float(t_compare)))
    if x <= 0:
        return None
    return f"{x}x speedup"


def consolidate_bottom_legend(
    fig,
    axs,
    *,
    fontsize: int = 11,
    ncol: int = 5,
) -> None:
    handles: list[object] = []
    labels: list[str] = []
    seen: set[str] = set()

    for ax in axs:
        handles_ax, labels_ax = ax.get_legend_handles_labels()
        for hi, li in zip(handles_ax, labels_ax, strict=False):
            if not li or li.startswith("_"):
                continue

            base = li.split(" (", 1)[0]
            if base.startswith("turbo-enn"):
                li2 = "turbo-enn"
            else:
                li2 = base
            if li2 in seen:
                continue
            seen.add(li2)
            handles.append(hi)
            labels.append(li2)

    for ax in axs:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    if not handles:
        return

    leg = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=int(ncol),
        frameon=False,
        fontsize=fontsize,
    )
    for handle in leg.legend_handles:
        handle.set_markersize(10)
        handle.set_linewidth(3.0)


def get_denoise_value(data_locator, problem: str) -> int:
    """Get num_denoise or num_denoise_passive from config based on problem type."""
    data_sets = data_locator._load(problem=problem)
    if not data_sets:
        return None

    dir_path = data_sets[0][1]
    config_path = Path(dir_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            if problem.endswith(":fn"):
                return config.get("num_denoise")
            else:
                return config.get("num_denoise_passive", config.get("num_denoise_eval", None))
    return None


def scan_experiment_configs(root: Path) -> tuple[set[str], set[str]]:
    """Return (env_tags, opt_names) found under an experiment root directory."""
    env_tags: set[str] = set()
    opt_names: set[str] = set()
    if not root.exists():
        return env_tags, opt_names

    for child in root.iterdir():
        if not child.is_dir():
            continue
        cfg = child / "config.json"
        if not cfg.exists():
            continue
        try:
            with open(cfg) as f:
                d = json.load(f)
        except Exception:
            continue
        env = d.get("env_tag") or d.get("env")
        opt = d.get("opt_name")
        if isinstance(env, str):
            env_tags.add(env)
        if isinstance(opt, str):
            opt_names.add(opt)
    return env_tags, opt_names


def infer_experiment_from_configs(results_path: str, exp_dir: str) -> dict:
    results_path, exp_dir = normalize_results_and_exp_dir(results_path, exp_dir)
    root = Path(results_path) / exp_dir
    if not root.exists():
        raise FileNotFoundError(str(root))

    cfgs: list[dict] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        p = child / "config.json"
        if not p.exists():
            continue
        try:
            with open(p) as f:
                cfgs.append(json.load(f))
        except Exception:
            continue

    if not cfgs:
        raise ValueError(f"No config.json files found under {str(root)!r}")

    env_tags = sorted({c.get("env_tag") or c.get("env") for c in cfgs if isinstance(c.get("env_tag") or c.get("env"), str)})
    opt_names = sorted({c.get("opt_name") for c in cfgs if isinstance(c.get("opt_name"), str)})

    def _uniq_int(key: str) -> int | None:
        xs = {c.get(key) for c in cfgs if isinstance(c.get(key), int)}
        if len(xs) == 1:
            return int(next(iter(xs)))
        return None

    out = {
        "results_path": results_path,
        "exp_dir": exp_dir,
        "env_tags": env_tags,
        "opt_names": opt_names,
        "num_arms": _uniq_int("num_arms"),
        "num_rounds": _uniq_int("num_rounds"),
        "num_reps": _uniq_int("num_reps"),
        "configs": cfgs,
    }
    return out


def infer_params_from_configs(
    results_path: str,
    exp_dir: str,
    *,
    problem_seq: str,
    problem_batch: str,
    opt_names: list[str],
) -> dict[str, int]:
    """Infer (num_rounds_seq, num_rounds_batch, num_arms_seq, num_arms_batch, num_reps) from config.json."""
    results_path, exp_dir = normalize_results_and_exp_dir(results_path, exp_dir)
    root = Path(results_path) / exp_dir

    rows = collect_config_rows(root, opt_names, include_opt_name=True)

    seq = [r for r in rows if r["env_tag"] == problem_seq]
    batch = [r for r in rows if r["env_tag"] == problem_batch]

    out: dict[str, int] = {}
    nr_seq = uniq_int([r["num_rounds"] for r in seq])
    nr_batch = uniq_int([r["num_rounds"] for r in batch])
    na_seq = uniq_int([r["num_arms"] for r in seq])
    na_batch = uniq_int([r["num_arms"] for r in batch])
    reps_seq = uniq_int([r["num_reps"] for r in seq])
    reps_batch = uniq_int([r["num_reps"] for r in batch])

    if nr_seq is not None:
        out["num_rounds_seq"] = nr_seq
    if nr_batch is not None:
        out["num_rounds_batch"] = nr_batch
    if na_seq is not None:
        out["num_arms_seq"] = na_seq
    if na_batch is not None:
        out["num_arms_batch"] = na_batch

    if reps_seq is not None and reps_batch is not None and reps_seq == reps_batch:
        out["num_reps"] = int(reps_seq)
    else:
        if reps_seq is not None:
            out["num_reps_seq"] = int(reps_seq)
        if reps_batch is not None:
            out["num_reps_batch"] = int(reps_batch)

    return out
