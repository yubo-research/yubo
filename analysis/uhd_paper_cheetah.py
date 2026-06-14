"""Load and plot HalfCheetah UHD-paper experiment traces (BO + UHD batches)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.data_io import TraceRecord, data_is_done, read_trace_jsonl
from analysis.uhd_paper_trace import best_so_far_array, sem_1d, summarize_curve
from experiments.batch_preps_rl_sweeps import (
    prep_cheetah_eggroll_s01,
    prep_cheetah_eggroll_s02,
    prep_cheetah_eggroll_s03,
    prep_cheetah_policy_optimizer,
)
from experiments.uhd_batch_preps import (
    prep_uhd_batch_cheetah,
    prep_uhd_batch_cheetah_simple,
)
from ops.uhd_batch_core import _config_hash, _experiment_dir

CHEETAH_FN_ENV_TAG = "cheetah:fn"
_EGGROLL_SIGMA_RE = re.compile(r"/sigma=([0-9.]+)/")
_EXTRA_EGGROLL_PREPS = (
    prep_cheetah_eggroll_s01,
    prep_cheetah_eggroll_s02,
    prep_cheetah_eggroll_s03,
)

UHD_BATCH_LABEL_ORDER: tuple[str, ...] = (
    "simple",
    "simple_be/flat/ucb",
    "simple_be/flat/mu",
)

BO_UHD_POLICIES: tuple[str, ...] = ("linear", "mlp-32-16")
BO_UHD_METHOD_ORDER: tuple[str, ...] = (
    "random",
    "turbo-enn-fit-ucb",
    "eggroll (sigma=0.05)",
    "eggroll (sigma=0.1)",
    "eggroll (sigma=0.2)",
    "eggroll (sigma=0.3)",
    "simple",
)

UHD_BATCH_COLORS: dict[str, str] = {
    "simple": "#1f77b4",
    "simple_be/flat/ucb": "#2ca02c",
    "simple_be/flat/mu": "#000000",
    "simple_be/hnsw/ucb": "#ff7f0e",
    "simple_be/hnsw/mu": "#ffbb78",
    "simple_be/hnsw_disk/ucb": "#d62728",
    "simple_be/hnsw_disk/mu": "#ff9896",
}

UHD_BATCH_LINESTYLES: dict[str, str] = {
    "simple": "-",
    "simple_be/flat/ucb": "-",
    "simple_be/flat/mu": "-",
    "simple_be/hnsw/ucb": "-",
    "simple_be/hnsw/mu": "--",
    "simple_be/hnsw_disk/ucb": "-",
    "simple_be/hnsw_disk/mu": "--",
}

UHD_BATCH_LINEWIDTHS: dict[str, float] = {
    "simple_be/flat/mu": 3.0,
}

BO_UHD_COLORS: dict[str, str] = {
    "random": "#7f7f7f",
    "turbo-enn-fit-ucb": "#2ca02c",
    "eggroll (sigma=0.05)": "#ff7f0e",
    "eggroll (sigma=0.1)": "#bcbd22",
    "eggroll (sigma=0.2)": "#9467bd",
    "eggroll (sigma=0.3)": "#8c564b",
    "simple": "#1f77b4",
}

BO_UHD_LINESTYLES: dict[str, str] = {
    "eggroll (sigma=0.1)": "--",
    "eggroll (sigma=0.2)": "-.",
    "eggroll (sigma=0.3)": ":",
}

DEFAULT_LINEWIDTH = 1.8

# LaTeX-safe legend strings (cell 1 enables text.usetex; avoid raw Unicode σ in labels).
LEGEND_LABELS: dict[str, str] = {}


def legend_label(method: str) -> str:
    if method in LEGEND_LABELS:
        return LEGEND_LABELS[method]
    m = re.fullmatch(r"eggroll \(sigma=([0-9.]+)\)", method)
    if m is not None:
        return rf"eggroll ($\sigma$={m.group(1)})"
    return method


@dataclass(frozen=True)
class CheetahExperimentPaths:
    yubo_root: Path

    @property
    def bo_policy_optimizer(self) -> Path:
        return self.yubo_root / "results" / "exp_cheetah_policy_optimizer"

    @property
    def bo_eggroll_s01(self) -> Path:
        return self.yubo_root / "results" / "exp_cheetah_eggroll_s01"

    @property
    def bo_eggroll_s02(self) -> Path:
        return self.yubo_root / "results" / "exp_cheetah_eggroll_s02"

    @property
    def bo_eggroll_s03(self) -> Path:
        return self.yubo_root / "results" / "exp_cheetah_eggroll_s03"

    @property
    def bo_extra_eggroll_dirs(self) -> tuple[Path, ...]:
        return (self.bo_eggroll_s01, self.bo_eggroll_s02, self.bo_eggroll_s03)

    @property
    def uhd(self) -> Path:
        return self.yubo_root / "results" / "uhd"


def label_uhd_config(cfg: dict) -> str:
    opt = cfg.get("optimizer", cfg.get("opt_name", "?"))
    if opt == "simple":
        return "simple"
    driver = cfg.get("be_enn_index_driver", "?")
    acq = cfg.get("be_acquisition", "?")
    return f"simple_be/{driver}/{acq}"


def normalize_bo_label(opt_name: str) -> str:
    if opt_name == "random":
        return "random"
    if opt_name == "turbo-enn-fit-ucb":
        return "turbo-enn-fit-ucb"
    if opt_name.startswith("eggroll"):
        m = _EGGROLL_SIGMA_RE.search(opt_name)
        if m is not None:
            return f"eggroll (sigma={m.group(1)})"
        return "eggroll"
    return opt_name


def load_traces_from_dir(traces_dir: Path) -> list[list[TraceRecord]]:
    traces: list[list[TraceRecord]] = []
    for trace_file in sorted(traces_dir.glob("*.jsonl")):
        if not data_is_done(str(trace_file)):
            continue
        records = read_trace_jsonl(str(trace_file))
        if records:
            traces.append(records)
    return traces


def load_uhd_batch_cheetah(results_dir: Path) -> tuple[list[dict], pd.DataFrame]:
    experiments: list[dict] = []
    rows: list[dict] = []
    for cfg, num_reps in prep_uhd_batch_cheetah(results_dir=str(results_dir)):
        label = label_uhd_config(cfg)
        cfg_hash = _config_hash(cfg)
        traces = load_traces_from_dir(Path(_experiment_dir(str(results_dir), cfg)) / "traces")
        experiments.append(
            {
                "hash": cfg_hash,
                "label": label,
                "config": cfg,
                "num_reps_expected": num_reps,
                "traces": traces,
            }
        )
        rows.append(
            {
                "label": label,
                "hash": cfg_hash,
                "reps_done": len(traces),
                "reps_expected": num_reps,
            }
        )
    summary = pd.DataFrame(rows).sort_values("label").reset_index(drop=True)
    return experiments, summary


def load_bo_cheetah_fn(results_dir: Path) -> list[dict]:
    experiments: list[dict] = []
    for cfg_path in sorted(results_dir.glob("*/config.json")):
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("env_tag") != CHEETAH_FN_ENV_TAG:
            continue
        traces = load_traces_from_dir(cfg_path.parent / "traces")
        if not traces:
            continue
        experiments.append(
            {
                "label": normalize_bo_label(str(cfg["opt_name"])),
                "policy_tag": cfg["policy_tag"],
                "config": cfg,
                "hash": cfg_path.parent.name,
                "traces": traces,
            }
        )
    return experiments


def load_uhd_cheetah_simple(results_dir: Path) -> list[dict]:
    experiments: list[dict] = []
    for cfg, _num_reps in prep_uhd_batch_cheetah_simple(results_dir=str(results_dir)):
        exp_dir = Path(_experiment_dir(str(results_dir), cfg))
        traces = load_traces_from_dir(exp_dir / "traces")
        if traces:
            experiments.append(
                {
                    "label": "simple",
                    "policy_tag": cfg["policy_tag"],
                    "config": cfg,
                    "hash": _config_hash(cfg),
                    "traces": traces,
                }
            )
    return experiments


def _extra_eggroll_coverage_rows(
    bo_experiments: Sequence[dict],
    loaded: set[tuple[str, str]],
) -> list[dict]:
    rows: list[dict] = []
    for prep_fn in _EXTRA_EGGROLL_PREPS:
        for cmd in prep_fn("results"):
            method = normalize_bo_label(cmd.opt_name)
            rows.append(
                {
                    "policy": cmd.policy_tag,
                    "method": method,
                    "expected_hash": Path(cmd.to_dir_name()).name,
                    "reps_done": next(
                        (len(exp["traces"]) for exp in bo_experiments if exp["policy_tag"] == cmd.policy_tag and exp["label"] == method),
                        0,
                    ),
                    "loaded": (cmd.policy_tag, method) in loaded,
                }
            )
    return rows


def bo_coverage_report(bo_experiments: Sequence[dict]) -> pd.DataFrame:
    loaded = {(exp["policy_tag"], exp["label"]) for exp in bo_experiments}
    rows: list[dict] = []
    for cmd in prep_cheetah_policy_optimizer("results"):
        method = normalize_bo_label(cmd.opt_name)
        rows.append(
            {
                "policy": cmd.policy_tag,
                "method": method,
                "expected_hash": Path(cmd.to_dir_name()).name,
                "reps_done": next(
                    (len(exp["traces"]) for exp in bo_experiments if exp["policy_tag"] == cmd.policy_tag and exp["label"] == method),
                    0,
                ),
                "loaded": (cmd.policy_tag, method) in loaded,
            }
        )
    rows.extend(_extra_eggroll_coverage_rows(bo_experiments, loaded))
    return pd.DataFrame(rows).sort_values(["policy", "method"]).reset_index(drop=True)


def load_cheetah_bo_uhd_comparison(
    paths: CheetahExperimentPaths,
) -> tuple[list[dict], list[dict], list[dict]]:
    bo_experiments = sum(
        (
            load_bo_cheetah_fn(paths.bo_policy_optimizer),
            *(load_bo_cheetah_fn(result_dir) for result_dir in paths.bo_extra_eggroll_dirs),
        ),
        [],
    )
    uhd_experiments = load_uhd_cheetah_simple(paths.uhd)
    return bo_experiments, uhd_experiments, bo_experiments + uhd_experiments


def loaded_traces_summary(experiments: Sequence[dict]) -> pd.DataFrame:
    rows = [
        {
            "policy": exp.get("policy_tag", ""),
            "method": exp["label"],
            "reps_done": len(exp["traces"]),
            "hash": exp["hash"],
        }
        for exp in experiments
    ]
    return pd.DataFrame(rows).sort_values(["policy", "method"]).reset_index(drop=True)


def final_best_summary(experiments: Sequence[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for exp in experiments:
        if not exp["traces"]:
            continue
        arr = best_so_far_array(exp["traces"])
        best = arr[:, -1]
        row = {
            "method": exp["label"],
            "reps": arr.shape[0],
            "best_mean": np.nanmean(best),
            "best_sem": sem_1d(best),
        }
        if "policy_tag" in exp:
            row["policy"] = exp["policy_tag"]
        else:
            row["label"] = exp["label"]
        rows.append(row)
    df = pd.DataFrame(rows)
    sort_cols = ["policy", "method"] if "policy" in df.columns else ["label"]
    return df.sort_values(sort_cols).reset_index(drop=True)


def _plot_curve_with_sem(
    ax,
    curve: dict,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
    linewidth: float = DEFAULT_LINEWIDTH,
    zorder: int = 2,
) -> None:
    x = curve["iterations"]
    ax.plot(
        x,
        curve["mean"],
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.fill_between(
        x,
        curve["mean"] - curve["sem"],
        curve["mean"] + curve["sem"],
        color=color,
        alpha=0.15,
        linewidth=0,
    )


def plot_uhd_batch_cheetah(
    experiments: Sequence[dict],
    *,
    label_order: Sequence[str] = UHD_BATCH_LABEL_ORDER,
    figsize: tuple[float, float] = (8.0, 4.5),
):
    ready = [e for e in experiments if e["traces"]]
    if not ready:
        raise ValueError("No completed traces to plot")

    curves = {exp["label"]: summarize_curve(exp["traces"]) for exp in ready}
    fig, ax = plt.subplots(figsize=figsize)
    for label in label_order:
        if label not in curves:
            continue
        curve = curves[label]
        _plot_curve_with_sem(
            ax,
            curve,
            color=UHD_BATCH_COLORS[label],
            label=label,
            linestyle=UHD_BATCH_LINESTYLES.get(label, "-"),
            linewidth=UHD_BATCH_LINEWIDTHS.get(label, DEFAULT_LINEWIDTH),
            zorder=3 if label == "simple_be/flat/mu" else 2,
        )

    ax.set_title("Best return so far: mean $\\pm$ SEM")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best return so far")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("HalfCheetah UHD batch\nnum_params = 1240", y=1.03)
    fig.tight_layout()
    return fig, final_best_summary(ready)


def plot_cheetah_bo_uhd_comparison(
    all_experiments: Sequence[dict],
    *,
    policies: Sequence[str] = BO_UHD_POLICIES,
    method_order: Sequence[str] = BO_UHD_METHOD_ORDER,
    figsize: tuple[float, float] = (12.0, 4.5),
):
    ready_by_policy = {policy: [e for e in all_experiments if e.get("policy_tag") == policy and e["traces"]] for policy in policies}
    if not any(ready_by_policy.values()):
        raise ValueError("No completed traces to plot")

    fig, axes = plt.subplots(1, len(policies), figsize=figsize, sharey=True)
    if len(policies) == 1:
        axes = [axes]

    for ax, policy in zip(axes, policies, strict=True):
        ready = ready_by_policy[policy]
        if not ready:
            ax.set_title(f"{policy}\n(no data)")
            continue

        curves = {exp["label"]: summarize_curve(exp["traces"]) for exp in ready}
        for method in method_order:
            if method not in curves:
                continue
            curve = curves[method]
            _plot_curve_with_sem(
                ax,
                curve,
                color=BO_UHD_COLORS[method],
                label=legend_label(method),
                linestyle=BO_UHD_LINESTYLES.get(method, "-"),
            )

        missing_methods = [m for m in method_order if m not in curves]
        subtitle = policy
        if missing_methods:
            subtitle += f"\n(missing: {', '.join(missing_methods)})"
        ax.set_title(subtitle)
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    axes[0].set_ylabel("Best return so far")
    fig.suptitle(
        r"HalfCheetah-v5 frozen noise ($\texttt{cheetah:fn}$)"
        "\n"
        r"BO: random / turbo-enn-fit-ucb / eggroll ($\sigma$=0.05–0.3) $\cdot$ UHD: simple",
        y=1.06,
    )
    fig.tight_layout()
    return fig, final_best_summary(all_experiments)
