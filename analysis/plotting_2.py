import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import analysis.data_sets as ds
import analysis.plotting as ap
from analysis.data_io import data_is_done
from analysis.data_locator import DataLocator


def _noise_label(problem: str) -> str:
    if problem.endswith(":fn"):
        return "Frozen noise"
    return "Natural noise"


def _speedup_x_label(
    cum_dt_prop_final_by_opt: dict[str, float] | None, problem: str
) -> str | None:
    if not cum_dt_prop_final_by_opt:
        return None

    baseline_candidates = ("turbo-one", "turbo-one-na", "turbo-one-f")
    baseline_opt = next(
        (o for o in baseline_candidates if o in cum_dt_prop_final_by_opt), None
    )
    if baseline_opt is None:
        return None

    compare_opt = "turbo-enn-p" if problem.endswith(":fn") else "turbo-enn-fit-ucb"
    if compare_opt not in cum_dt_prop_final_by_opt:
        return None

    t_baseline = cum_dt_prop_final_by_opt.get(baseline_opt, None)
    t_compare = cum_dt_prop_final_by_opt.get(compare_opt, None)
    if (
        t_baseline is None
        or t_compare is None
        or not np.isfinite(t_baseline)
        or not np.isfinite(t_compare)
        or t_compare <= 0
    ):
        return None

    x = int(round(float(t_baseline) / float(t_compare)))
    if x <= 0:
        return None
    return f"{x}x speedup"


def _consolidate_bottom_legend(
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


def _get_denoise_value(data_locator: DataLocator, problem: str) -> int:
    """Get num_denoise or num_denoise_passive from config based on problem type."""
    data_sets = data_locator._load(problem=problem)
    if not data_sets:
        return None

    # Read config from first matching directory
    dir_path = data_sets[0][1]
    config_path = Path(dir_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            if problem.endswith(":fn"):
                return config.get("num_denoise")
            else:
                return config.get(
                    "num_denoise_passive", config.get("num_denoise_eval", None)
                )
    return None


def _normalize_results_and_exp_dir(results_path: str, exp_dir: str) -> tuple[str, str]:
    """
    Normalize (results_path, exp_dir) to what DataLocator expects.

    DataLocator assumes root = Path(results_path) / exp_dir.
    This helper makes common caller variants work:
    - results_path is repo root (contains a "results/" subdir)
    - exp_dir is "results/<exp>" while results_path already points at ".../results"
    - exp_dir is an absolute/relative path to the experiment directory
    - results_path itself points at the experiment directory
    """
    rp = Path(results_path).expanduser()
    ed = Path(exp_dir).expanduser()

    # Case: results_path is repo root and has results/<exp_dir>
    if (rp / "results" / exp_dir).exists():
        return str(rp / "results"), exp_dir

    # Case: exp_dir includes "results/<exp>" while results_path already ends with ".../results"
    if rp.name == "results" and exp_dir.startswith("results/"):
        exp_dir = exp_dir[len("results/") :]
        return str(rp), exp_dir

    # Case: exp_dir is a path to the experiment directory
    if ed.exists() and ed.is_dir():
        return str(ed.parent), ed.name

    # Case: results_path points directly at the experiment directory
    if rp.exists() and rp.is_dir() and rp.name == exp_dir:
        return str(rp.parent), rp.name

    return str(rp), exp_dir


def _scan_experiment_configs(root: Path) -> tuple[set[str], set[str]]:
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
    results_path, exp_dir = _normalize_results_and_exp_dir(results_path, exp_dir)
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

    env_tags = sorted(
        {
            c.get("env_tag") or c.get("env")
            for c in cfgs
            if isinstance(c.get("env_tag") or c.get("env"), str)
        }
    )
    opt_names = sorted(
        {c.get("opt_name") for c in cfgs if isinstance(c.get("opt_name"), str)}
    )

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


def _infer_params_from_configs(
    results_path: str,
    exp_dir: str,
    *,
    problem_seq: str,
    problem_batch: str,
    opt_names: list[str],
) -> dict[str, int]:
    """Infer (num_rounds_seq, num_rounds_batch, num_arms_seq, num_arms_batch, num_reps) from config.json."""
    results_path, exp_dir = _normalize_results_and_exp_dir(results_path, exp_dir)
    root = Path(results_path) / exp_dir

    rows: list[dict] = []
    for child in root.iterdir() if root.exists() else []:
        if not child.is_dir():
            continue
        cfg = child / "config.json"
        if not cfg.exists():
            continue
        try:
            with open(cfg) as f:
                c = json.load(f)
        except Exception:
            continue
        env_tag = c.get("env_tag") or c.get("env")
        opt = c.get("opt_name")
        if not isinstance(env_tag, str) or not isinstance(opt, str):
            continue
        if opt not in opt_names:
            continue
        rows.append(
            {
                "env_tag": env_tag,
                "opt_name": opt,
                "num_arms": c.get("num_arms"),
                "num_rounds": c.get("num_rounds"),
                "num_reps": c.get("num_reps"),
            }
        )

    def _uniq_int(vals: list) -> int | None:
        xs = {v for v in vals if isinstance(v, int)}
        if len(xs) == 1:
            return next(iter(xs))
        return None

    seq = [r for r in rows if r["env_tag"] == problem_seq]
    batch = [r for r in rows if r["env_tag"] == problem_batch]

    out: dict[str, int] = {}
    nr_seq = _uniq_int([r["num_rounds"] for r in seq])
    nr_batch = _uniq_int([r["num_rounds"] for r in batch])
    na_seq = _uniq_int([r["num_arms"] for r in seq])
    na_batch = _uniq_int([r["num_arms"] for r in batch])
    reps_seq = _uniq_int([r["num_reps"] for r in seq])
    reps_batch = _uniq_int([r["num_reps"] for r in batch])

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


def load_rl_traces(
    results_path: str,
    exp_dir: str,
    opt_names: list[str],
    num_arms: int,
    num_rounds: int,
    num_reps: int | None,
    problem: str,
    key: str = "rreturn",
):
    results_path, exp_dir = _normalize_results_and_exp_dir(results_path, exp_dir)
    root = Path(results_path) / exp_dir

    data_locator = DataLocator(
        results_path,
        exp_dir,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
        opt_names=opt_names,
        problems={problem},
        problems_exact=True,
        key=key,
    )

    # If no problems match *after applying filters*, provide a targeted diagnosis.
    problems_found = data_locator.problems()
    if not problems_found:
        env_tags, opt_names_found = _scan_experiment_configs(root)
        env_tags_sorted = sorted(env_tags)
        opt_names_sorted = sorted(opt_names_found)

        root_exists = root.exists()
        root_hint = ""
        if root_exists:
            try:
                entries = sorted([p.name for p in root.iterdir()])[:5]
                root_hint = f" root_entries[:5]={entries!r}"
            except Exception:
                root_hint = ""

        # Check whether the requested problem appears at all in this experiment directory.
        problem_present = any(problem in env for env in env_tags_sorted)
        if not problem_present:
            raise ValueError(
                "No matching problems found. "
                f"results_path={results_path!r}, exp_dir={exp_dir!r}, root={str(root)!r}, root_exists={root_exists}, "
                f"requested problem={problem!r}.{root_hint} "
                f"available_env_tags={env_tags_sorted!r} "
                "This usually means you're pointing at the wrong experiment directory for that problem."
            )

        # Otherwise, the problem exists, but your filters (opt_names / num_arms / num_rounds / num_reps) excluded all runs.
        raise ValueError(
            "No matching runs after applying filters. "
            f"results_path={results_path!r}, exp_dir={exp_dir!r}, root={str(root)!r}, root_exists={root_exists}, "
            f"requested problem={problem!r}, num_arms={num_arms}, num_rounds={num_rounds}, num_reps={num_reps}, "
            f"requested opt_names={opt_names!r}. "
            f"available_env_tags={env_tags_sorted!r} available_opt_names={opt_names_sorted!r}"
        )
    traces = ds.load_multiple_traces(data_locator)
    return data_locator, traces


def _count_done_reps(trace_dir: str) -> int:
    p = Path(trace_dir)
    traces_subdir = p / "traces"
    if traces_subdir.exists():
        p = traces_subdir
    jsonl_files = sorted(p.glob("*.jsonl"))
    if jsonl_files:
        return int(sum(1 for fn in jsonl_files if data_is_done(str(fn))))
    n = 0
    for fn in sorted(p.iterdir()):
        if not fn.is_file():
            continue
        if fn.name.endswith(".done") or fn.name.endswith(".jsonl"):
            continue
        if data_is_done(str(fn)):
            n += 1
    return int(n)


def _print_dataset_summary(
    results_path: str,
    exp_dir: str,
    *,
    problem: str,
    opt_names: list[str],
    num_arms: int,
    num_rounds: int,
    num_reps: int | None,
):
    data_locator = DataLocator(
        results_path,
        exp_dir,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
        opt_names=opt_names,
        problems={problem},
        problems_exact=True,
        key="rreturn",
    )
    for opt in data_locator.optimizers():
        paths = data_locator(problem, opt)
        if not paths:
            continue
        trace_dir = paths[0]
        cfg_reps = None
        cfg_path = Path(trace_dir) / "config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    cfg_reps = json.load(f).get("num_reps")
            except Exception:
                cfg_reps = None
        reps_done = _count_done_reps(trace_dir)
        root = Path(trace_dir).name
        print(
            f"PLOT: env={problem} opt={opt} arms={num_arms} rounds={num_rounds} reps_done={reps_done} reps_cfg={cfg_reps} dir={root}"
        )


def _mean_final_by_optimizer(
    data_locator: DataLocator, traces: np.ndarray
) -> dict[str, float]:
    """Return {opt_name: mean(final_value_over_reps)} for a single-problem trace tensor."""
    optimizers = data_locator.optimizers()
    z = traces.squeeze(0)  # [n_opt, n_rep, n_round]
    if z.ndim != 3 or z.shape[2] == 0:
        raise ValueError(
            f"Empty traces: shape={getattr(z, 'shape', None)} for key={getattr(data_locator, 'key', None)}"
        )
    out: dict[str, float] = {}
    for i_opt, opt_name in enumerate(optimizers):
        y_final = z[i_opt, :, -1]
        # Support masked arrays
        try:
            out[opt_name] = float(np.ma.mean(y_final))
        except Exception:
            out[opt_name] = float(np.mean(np.asarray(y_final, dtype=float)))
    return out


def _median_final_by_optimizer(
    data_locator: DataLocator, traces: np.ndarray
) -> dict[str, float]:
    """Return {opt_name: median(final_value_over_reps)} for a single-problem trace tensor."""
    optimizers = data_locator.optimizers()
    z = traces.squeeze(0)  # [n_opt, n_rep, n_round]
    if z.ndim != 3 or z.shape[2] == 0:
        raise ValueError(
            f"Empty traces: shape={getattr(z, 'shape', None)} for key={getattr(data_locator, 'key', None)}"
        )
    out: dict[str, float] = {}
    for i_opt, opt_name in enumerate(optimizers):
        y_final = z[i_opt, :, -1]
        try:
            out[opt_name] = float(np.ma.median(y_final))
        except Exception:
            out[opt_name] = float(np.median(np.asarray(y_final, dtype=float)))
    return out


def _normalized_ranks_0_1(scores_1d: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores_1d, dtype=float)
    bad = ~np.isfinite(scores)
    if np.any(bad):
        scores = scores.copy()
        scores[bad] = -np.inf

    n = int(scores.shape[0])
    if n <= 1:
        return np.ones_like(scores, dtype=float)

    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    for i_rank, i_opt in enumerate(order, start=1):
        ranks[i_opt] = float(i_rank)

    return 1.0 - (ranks - 1.0) / float(n - 1)


def _mean_normalized_rank_score_by_optimizer(
    data_locator: DataLocator,
    traces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    optimizers = data_locator.optimizers()
    z = traces.squeeze(0)  # [n_opt, n_rep, n_round]
    n_opt = int(z.shape[0])
    n_rep = int(z.shape[1])
    n_round = int(z.shape[2])

    scores_by_rep = np.full((n_opt, n_rep), np.nan, dtype=float)
    for i_rep in range(n_rep):
        norm_ranks = np.full((n_opt, n_round), np.nan, dtype=float)
        for t in range(n_round):
            norm_ranks[:, t] = _normalized_ranks_0_1(z[:, i_rep, t])
        scores_by_rep[:, i_rep] = np.nanmean(norm_ranks, axis=1)

    means = np.nanmean(scores_by_rep, axis=1)
    stes = np.nanstd(scores_by_rep, axis=1) / np.sqrt(float(n_rep))
    assert means.shape == (len(optimizers),)
    assert stes.shape == (len(optimizers),)
    return means, stes


def _cum_dt_prop_from_dt_prop_traces(dt_prop_traces: np.ndarray) -> np.ndarray:
    """
    Convert dt_prop traces into cumulative dt_prop traces.

    Input shape: [1, n_opt, n_rep, n_round] (masked or ndarray).
    Output has same shape, with cumsum over rounds.
    """
    z = dt_prop_traces.squeeze(0)  # [n_opt, n_rep, n_round]
    if np.ma.isMaskedArray(z):
        z_cum = np.ma.cumsum(z, axis=-1)
    else:
        z_cum = np.cumsum(z, axis=-1)
    return np.expand_dims(z_cum, axis=0)


def _print_cum_dt_prop(
    cum_dt_prop_final_by_opt: dict[str, float] | None,
    opt_order: list[str] | None,
    *,
    header: str,
) -> None:
    if not cum_dt_prop_final_by_opt:
        return
    order = opt_order or sorted(cum_dt_prop_final_by_opt.keys())
    items = [
        f"\\texttt{{{o}}} {cum_dt_prop_final_by_opt[o]:.1f}s"
        for o in order
        if o in cum_dt_prop_final_by_opt
    ]
    if not items:
        return
    print(header)
    print("  " + ", ".join(items))


def _load_cum_dt_prop(
    results_path: str,
    exp_dir: str,
    opt_names: list[str],
    *,
    num_arms: int,
    num_rounds: int,
    num_reps: int,
    problem: str,
) -> tuple[DataLocator, np.ndarray]:
    """
    Load dt_prop and compute cum_dt_prop = cumsum(dt_prop) over the run.
    """
    data_locator_dt, traces_dt = load_rl_traces(
        results_path,
        exp_dir,
        opt_names,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
        problem=problem,
        key="dt_prop",
    )
    traces_cum = _cum_dt_prop_from_dt_prop_traces(traces_dt)
    return data_locator_dt, traces_cum


def plot_learning_curves(
    ax,
    data_locator: DataLocator,
    traces: np.ndarray,
    num_arms: int = 1,
    title: str = None,
    xlabel: str = "N",
    ylabel: str = "$y_{best}$",
    markersize: int = 5,
    cum_dt_prop_final_by_opt: dict[str, float] | None = None,
    x_start: int = 1,
    opt_names_all: list[str] | None = None,
    show_title: bool = True,
):
    optimizers = data_locator.optimizers()
    z = traces.squeeze(0)

    for i_opt, opt_name in enumerate(optimizers):
        y = z[i_opt, ...]
        x = num_arms * (int(x_start) + np.arange(y.shape[1]))
        style_idx = (
            opt_names_all.index(opt_name)
            if opt_names_all and opt_name in opt_names_all
            else i_opt
        )
        if opt_name.startswith("turbo-enn"):
            color = "#333333"
            marker = "s"
        else:
            color = ap.colors[style_idx]
            marker = ap.markers[style_idx]
        label = opt_name
        if (
            cum_dt_prop_final_by_opt is not None
            and opt_name in cum_dt_prop_final_by_opt
        ):
            label = f"{opt_name} ({cum_dt_prop_final_by_opt[opt_name]:.1f}s)"
        ap.filled_err(
            x=x,
            ys=y,
            ax=ax,
            se=True,
            alpha=0.25,
            color=color,
            color_line=color,
            label=label,
            marker=marker,
            max_markers=10,
            markersize=markersize,
        )

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    if show_title and title:
        ax.set_title(title, fontsize=16)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)


def plot_final_performance(
    ax,
    data_locator: DataLocator,
    traces: np.ndarray,
    title: str = None,
    ylabel: str = "Mean normalized rank",
    opt_names_all: list[str] | None = None,
    show_title: bool = True,
):
    optimizers = data_locator.optimizers()
    means, stes = _mean_normalized_rank_score_by_optimizer(data_locator, traces)

    colors = []
    for opt_name in optimizers:
        if opt_name.startswith("turbo-enn"):
            colors.append("#333333")
        elif opt_names_all and opt_name in opt_names_all:
            colors.append(ap.colors[opt_names_all.index(opt_name)])
        else:
            colors.append(ap.colors[optimizers.index(opt_name)])

    x_pos = np.arange(len(optimizers))
    ax.bar(
        x_pos,
        means,
        yerr=2 * stes,
        capsize=5,
        color=colors,
        alpha=0.8,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(optimizers, rotation=45, ha="right")
    ax.set_ylabel(ylabel, fontsize=12)
    if show_title and title:
        ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(-0.05, 1.05)


def plot_rl_experiment(
    results_path: str,
    exp_dir: str,
    opt_names: list[str],
    num_arms: int,
    num_rounds: int,
    num_reps: int,
    problem: str,
    title: str = None,
    figsize: tuple = (10, 6),
):
    data_locator, traces = load_rl_traces(
        results_path, exp_dir, opt_names, num_arms, num_rounds, num_reps, problem
    )
    cum_dt_prop_final_by_opt = None
    try:
        data_locator_dt, traces_cum = _load_cum_dt_prop(
            results_path,
            exp_dir,
            opt_names,
            num_arms=num_arms,
            num_rounds=num_rounds,
            num_reps=num_reps,
            problem=problem,
        )
        cum_dt_prop_final_by_opt = _mean_final_by_optimizer(data_locator_dt, traces_cum)
    except ValueError:
        cum_dt_prop_final_by_opt = None

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_learning_curves(
        ax,
        data_locator,
        traces,
        num_arms=num_arms,
        title=title,
        cum_dt_prop_final_by_opt=cum_dt_prop_final_by_opt,
        opt_names_all=opt_names,
    )
    plt.tight_layout()
    return fig, ax, data_locator, traces


def _best_so_far(traces: np.ndarray) -> np.ndarray:
    z = traces
    if np.ma.isMaskedArray(z):
        z2 = z.filled(np.nan)
        z2 = np.maximum.accumulate(z2, axis=-1)
        return np.ma.masked_invalid(z2)
    z2 = np.asarray(z, dtype=float)
    z2 = np.maximum.accumulate(z2, axis=-1)
    return z2


def _cum_time_from_dt(dt_prop: np.ndarray, dt_eval: np.ndarray) -> np.ndarray:
    z = dt_prop + dt_eval
    if np.ma.isMaskedArray(z):
        return np.ma.cumsum(z, axis=-1)
    return np.cumsum(np.asarray(z, dtype=float), axis=-1)


def _interp_1d(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if np.sum(ok) < 2:
        return np.full_like(xq, np.nan, dtype=float)
    xo = x[ok]
    yo = y[ok]
    order = np.argsort(xo, kind="mergesort")
    xo = xo[order]
    yo = yo[order]
    xo, uniq = np.unique(xo, return_index=True)
    yo = yo[uniq]
    if xo.shape[0] < 2:
        return np.full_like(xq, np.nan, dtype=float)
    xq = np.asarray(xq, dtype=float)
    xq_clip = np.clip(xq, xo[0], xo[-1])
    return np.interp(xq_clip, xo, yo)


def plot_rl_experiment_vs_time(
    results_path: str,
    exp_dir: str,
    opt_names: list[str],
    num_arms: int,
    num_rounds: int,
    num_reps: int,
    problem: str,
    title: str | None = None,
    figsize: tuple = (10, 6),
    n_grid: int = 200,
):
    data_locator_ret, traces_ret = load_rl_traces(
        results_path,
        exp_dir,
        opt_names,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
        problem=problem,
        key="rreturn",
    )
    _, traces_dt_prop = load_rl_traces(
        results_path,
        exp_dir,
        opt_names,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
        problem=problem,
        key="dt_prop",
    )
    _, traces_dt_eval = load_rl_traces(
        results_path,
        exp_dir,
        opt_names,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
        problem=problem,
        key="dt_eval",
    )

    y = _best_so_far(traces_ret.squeeze(0))
    t = _cum_time_from_dt(traces_dt_prop.squeeze(0), traces_dt_eval.squeeze(0))

    optimizers = data_locator_ret.optimizers()
    n_opt = int(y.shape[0])
    n_rep = int(y.shape[1])
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i_opt in range(n_opt):
        color = ap.colors[i_opt]
        marker = ap.markers[i_opt]
        label = optimizers[i_opt]

        ti = t[i_opt, ...]
        yi = y[i_opt, ...]

        if n_rep == 1:
            x = np.asarray(ti[0], dtype=float)
            yy = np.asarray(yi[0], dtype=float)
            ok = np.isfinite(x) & np.isfinite(yy)
            ax.plot(
                x[ok],
                yy[ok],
                color=color,
                label=label,
                marker=marker,
                markevery=max(1, int(np.sum(ok) / 10)),
            )
            continue

        t_ends = []
        for r in range(n_rep):
            xr = np.asarray(ti[r], dtype=float)
            ok = np.isfinite(xr)
            if np.any(ok):
                t_ends.append(float(np.nanmax(xr[ok])))
        if not t_ends:
            continue
        t_max = float(np.nanmin(t_ends))
        if not np.isfinite(t_max) or t_max <= 0:
            continue
        xq = np.linspace(0.0, t_max, int(n_grid))

        yq = np.full((n_rep, xq.shape[0]), np.nan, dtype=float)
        for r in range(n_rep):
            xr = np.asarray(ti[r], dtype=float)
            yr = np.asarray(yi[r], dtype=float)
            yq[r, :] = _interp_1d(xr, yr, xq)

        mu = np.nanmean(yq, axis=0)
        se = np.nanstd(yq, axis=0) / np.sqrt(float(n_rep))
        ax.plot(
            xq,
            mu,
            color=color,
            label=label,
            marker=marker,
            markevery=max(1, int(xq.shape[0] / 10)),
        )
        ax.fill_between(xq, mu - se, mu + se, color=color, alpha=0.25)

    ax.set_xlabel("Cumulative time (s)", fontsize=12)
    ax.set_ylabel("Return (best so far)", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax, data_locator_ret, traces_ret, t


def plot_rl_experiment_vs_time_auto(
    results_path: str,
    exp_dir: str,
    *,
    problem: str | None = None,
    opt_names: list[str] | None = None,
    num_arms: int | None = None,
    num_rounds: int | None = None,
    num_reps: int | None = None,
    title: str | None = None,
    figsize: tuple = (10, 6),
    n_grid: int = 200,
):
    info = infer_experiment_from_configs(results_path, exp_dir)
    results_path = info["results_path"]
    exp_dir = info["exp_dir"]

    if opt_names is None:
        opt_names = info["opt_names"]
    if not opt_names:
        raise ValueError(
            f"No opt_names found for results_path={results_path!r}, exp_dir={exp_dir!r}"
        )

    if problem is None:
        env_tags = info["env_tags"]
        if len(env_tags) != 1:
            raise ValueError(
                f"Multiple env_tags found {env_tags!r}; pass problem= explicitly"
            )
        problem = env_tags[0]

    if num_arms is None:
        num_arms = info["num_arms"]
    if num_rounds is None:
        num_rounds = info["num_rounds"]
    if num_reps is None:
        num_reps = info["num_reps"]

    missing = [
        k
        for k, v in [
            ("num_arms", num_arms),
            ("num_rounds", num_rounds),
            ("num_reps", num_reps),
        ]
        if v is None
    ]
    if missing:
        raise ValueError(f"Couldn't infer {missing} from config.json; pass explicitly")

    return plot_rl_experiment_vs_time(
        results_path,
        exp_dir,
        opt_names,
        num_arms=int(num_arms),
        num_rounds=int(num_rounds),
        num_reps=int(num_reps),
        problem=problem,
        title=title,
        figsize=figsize,
        n_grid=n_grid,
    )


def plot_rl_comparison(
    results_path: str,
    exp_dir: str,
    opt_names_seq: list[str],
    opt_names_batch: list[str],
    problem_seq: str,
    problem_batch: str,
    num_reps: int | None = None,
    num_rounds_seq: int = 100,
    num_rounds_batch: int = 30,
    num_arms_seq: int = 1,
    num_arms_batch: int = 50,
    suptitle: str = None,
    figsize: tuple = (12, 5),
    cum_dt_prop: bool = False,
    opt_names_all: list[str] | None = None,
    show_titles: bool = True,
    print_titles: bool = False,
):
    data_locator_seq, traces_seq = load_rl_traces(
        results_path,
        exp_dir,
        opt_names_seq,
        num_arms=num_arms_seq,
        num_rounds=num_rounds_seq,
        num_reps=num_reps,
        problem=problem_seq,
    )
    cum_dt_prop_seq = None
    data_locator_seq_dt = None
    traces_seq_dt = None
    try:
        data_locator_seq_dt, traces_seq_dt = load_rl_traces(
            results_path,
            exp_dir,
            opt_names_seq,
            num_arms=num_arms_seq,
            num_rounds=num_rounds_seq,
            num_reps=num_reps,
            problem=problem_seq,
            key="dt_prop",
        )
        traces_seq_cum = _cum_dt_prop_from_dt_prop_traces(traces_seq_dt)
        cum_dt_prop_seq = _median_final_by_optimizer(
            data_locator_seq_dt, traces_seq_cum
        )
    except ValueError:
        cum_dt_prop_seq = None
        data_locator_seq_dt = None
        traces_seq_dt = None

    data_locator_batch = None
    traces_batch = None
    try:
        data_locator_batch, traces_batch = load_rl_traces(
            results_path,
            exp_dir,
            opt_names_batch,
            num_arms=num_arms_batch,
            num_rounds=num_rounds_batch,
            num_reps=num_reps,
            problem=problem_batch,
        )
    except ValueError:
        data_locator_batch = None
        traces_batch = None
    cum_dt_prop_batch = None
    data_locator_batch_dt = None
    traces_batch_dt = None
    try:
        if data_locator_batch is not None:
            data_locator_batch_dt, traces_batch_dt = load_rl_traces(
                results_path,
                exp_dir,
                opt_names_batch,
                num_arms=num_arms_batch,
                num_rounds=num_rounds_batch,
                num_reps=num_reps,
                problem=problem_batch,
                key="dt_prop",
            )
            traces_batch_cum = _cum_dt_prop_from_dt_prop_traces(traces_batch_dt)
            cum_dt_prop_batch = _median_final_by_optimizer(
                data_locator_batch_dt, traces_batch_cum
            )
    except ValueError:
        cum_dt_prop_batch = None
        data_locator_batch_dt = None
        traces_batch_dt = None

    _print_cum_dt_prop(
        cum_dt_prop_seq,
        opt_names_all if opt_names_all else opt_names_seq,
        header=f"cumulative proposal times ({problem_seq})",
    )
    _print_cum_dt_prop(
        cum_dt_prop_batch,
        opt_names_all if opt_names_all else opt_names_batch,
        header=f"cumulative proposal times ({problem_batch})",
    )

    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True)

    noise_seq = _noise_label(problem_seq)
    denoise_seq = _get_denoise_value(data_locator_seq, problem_seq)
    speedup_seq = _speedup_x_label(cum_dt_prop_seq, problem_seq)
    line1_seq = f"{noise_seq}, {speedup_seq}" if speedup_seq else noise_seq
    parts_seq = [f"num_arms = {num_arms_seq}"]
    if denoise_seq is not None:
        denoise_key_seq = (
            "num_denoise_obs" if problem_seq.endswith(":fn") else "num_denoise_passive"
        )
        parts_seq.append(f"{denoise_key_seq} = {denoise_seq}")
    title_seq = f"{line1_seq}\n{', '.join(parts_seq)}"
    if print_titles:
        print(title_seq)
    plot_learning_curves(
        axs[0],
        data_locator_seq,
        traces_seq,
        num_arms=num_arms_seq,
        title=title_seq,
        cum_dt_prop_final_by_opt=cum_dt_prop_seq,
        opt_names_all=opt_names_all if opt_names_all else opt_names_seq,
        show_title=show_titles,
    )

    if data_locator_batch is not None and traces_batch is not None:
        noise_batch = _noise_label(problem_batch)
        denoise_batch = _get_denoise_value(data_locator_batch, problem_batch)
        speedup_batch = _speedup_x_label(cum_dt_prop_batch, problem_batch)
        line1_batch = (
            f"{noise_batch}, {speedup_batch}" if speedup_batch else noise_batch
        )
        parts_batch = [f"num_arms = {num_arms_batch}"]
        if denoise_batch is not None:
            denoise_key_batch = (
                "num_denoise_obs"
                if problem_batch.endswith(":fn")
                else "num_denoise_passive"
            )
            parts_batch.append(f"{denoise_key_batch} = {denoise_batch}")
        title_batch = f"{line1_batch}\n{', '.join(parts_batch)}"
        if print_titles:
            print(title_batch)
        plot_learning_curves(
            axs[1],
            data_locator_batch,
            traces_batch,
            num_arms=num_arms_batch,
            title=title_batch,
            cum_dt_prop_final_by_opt=cum_dt_prop_batch,
            opt_names_all=opt_names_all if opt_names_all else opt_names_batch,
            show_title=show_titles,
        )
    else:
        axs[1].axis("off")

    axs[0].set_xlabel("N", fontsize=16)
    if axs[1].axison:
        axs[1].set_xlabel("N", fontsize=16)
        axs[1].set_ylabel("")
        axs[1].tick_params(axis="y", labelleft=False, labelsize=16)

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=1.02)
    _consolidate_bottom_legend(
        fig,
        axs,
        fontsize=16,
        ncol=6,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))

    return fig, axs, (data_locator_seq, traces_seq), (data_locator_batch, traces_batch)


def plot_rl_final_comparison(
    results_path: str,
    exp_dir: str,
    opt_names_seq: list[str],
    opt_names_batch: list[str],
    problem_seq: str,
    problem_batch: str,
    num_reps: int | None = None,
    num_rounds_seq: int = 100,
    num_rounds_batch: int = 30,
    num_arms_seq: int = 1,
    num_arms_batch: int = 50,
    suptitle: str = None,
    figsize: tuple = (14, 5),
    opt_names_all: list[str] | None = None,
    show_titles: bool = True,
    print_titles: bool = False,
):
    data_locator_seq, traces_seq = load_rl_traces(
        results_path,
        exp_dir,
        opt_names_seq,
        num_arms=num_arms_seq,
        num_rounds=num_rounds_seq,
        num_reps=num_reps,
        problem=problem_seq,
    )

    cum_dt_prop_seq = None
    try:
        data_locator_seq_dt, traces_seq_dt = load_rl_traces(
            results_path,
            exp_dir,
            opt_names_seq,
            num_arms=num_arms_seq,
            num_rounds=num_rounds_seq,
            num_reps=num_reps,
            problem=problem_seq,
            key="dt_prop",
        )
        traces_seq_cum = _cum_dt_prop_from_dt_prop_traces(traces_seq_dt)
        cum_dt_prop_seq = _median_final_by_optimizer(
            data_locator_seq_dt, traces_seq_cum
        )
    except ValueError:
        cum_dt_prop_seq = None

    data_locator_batch = None
    traces_batch = None
    try:
        data_locator_batch, traces_batch = load_rl_traces(
            results_path,
            exp_dir,
            opt_names_batch,
            num_arms=num_arms_batch,
            num_rounds=num_rounds_batch,
            num_reps=num_reps,
            problem=problem_batch,
        )
    except ValueError:
        data_locator_batch = None
        traces_batch = None

    cum_dt_prop_batch = None
    try:
        if data_locator_batch is not None:
            data_locator_batch_dt, traces_batch_dt = load_rl_traces(
                results_path,
                exp_dir,
                opt_names_batch,
                num_arms=num_arms_batch,
                num_rounds=num_rounds_batch,
                num_reps=num_reps,
                problem=problem_batch,
                key="dt_prop",
            )
            traces_batch_cum = _cum_dt_prop_from_dt_prop_traces(traces_batch_dt)
            cum_dt_prop_batch = _median_final_by_optimizer(
                data_locator_batch_dt, traces_batch_cum
            )
    except ValueError:
        cum_dt_prop_batch = None

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    noise_seq = _noise_label(problem_seq)
    denoise_seq = _get_denoise_value(data_locator_seq, problem_seq)
    speedup_seq = _speedup_x_label(cum_dt_prop_seq, problem_seq)
    line1_seq = f"{noise_seq}, {speedup_seq}" if speedup_seq else noise_seq
    parts_seq = [f"num_arms = {num_arms_seq}"]
    if denoise_seq is not None:
        denoise_key_seq = (
            "num_denoise" if problem_seq.endswith(":fn") else "num_denoise_passive"
        )
        parts_seq.append(f"{denoise_key_seq} = {denoise_seq}")
    title_seq = f"{line1_seq}\n{', '.join(parts_seq)}"
    if print_titles:
        print(title_seq)
    plot_final_performance(
        axs[0],
        data_locator_seq,
        traces_seq,
        title=title_seq,
        opt_names_all=opt_names_all if opt_names_all else opt_names_seq,
        show_title=show_titles,
    )

    if data_locator_batch is not None and traces_batch is not None:
        noise_batch = _noise_label(problem_batch)
        denoise_batch = _get_denoise_value(data_locator_batch, problem_batch)
        speedup_batch = _speedup_x_label(cum_dt_prop_batch, problem_batch)
        line1_batch = (
            f"{noise_batch}, {speedup_batch}" if speedup_batch else noise_batch
        )
        parts_batch = [f"num_arms = {num_arms_batch}"]
        if denoise_batch is not None:
            denoise_key_batch = (
                "num_denoise"
                if problem_batch.endswith(":fn")
                else "num_denoise_passive"
            )
            parts_batch.append(f"{denoise_key_batch} = {denoise_batch}")
        title_batch = f"{line1_batch}\n{', '.join(parts_batch)}"
        if print_titles:
            print(title_batch)
        plot_final_performance(
            axs[1],
            data_locator_batch,
            traces_batch,
            title=title_batch,
            opt_names_all=opt_names_all if opt_names_all else opt_names_batch,
            show_title=show_titles,
        )
    else:
        axs[1].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=1.02)
    plt.tight_layout()

    return fig, axs, (data_locator_seq, traces_seq), (data_locator_batch, traces_batch)


_long_names = {
    "tlunar": "LunarLander-v3",
    "push": "Push-v3",
    "hop": "Hopper-v5",
    "bw-heur": "BipedalWalker-v3",
    "dna": "LASSO-DNA",
}


def plot_results(
    results_path: str,
    exp_dir: str,
    opt_names: list[str],
    problem: str,
    num_reps: int | None = None,
    num_rounds_seq: int | None = None,
    num_rounds_batch: int | None = None,
    num_arms_seq: int | None = None,
    num_arms_batch: int | None = None,
    exclude_seq: list[str] | None = None,
    exclude_batch: list[str] | None = None,
):
    problem_name = _long_names.get(problem, problem)
    problem_seq = problem
    problem_batch = f"{problem}:fn"

    opt_names_seq = [o for o in opt_names if o not in (exclude_seq or [])]
    opt_names_batch = [o for o in opt_names if o not in (exclude_batch or [])]

    inferred = _infer_params_from_configs(
        results_path,
        exp_dir,
        problem_seq=problem_seq,
        problem_batch=problem_batch,
        opt_names=opt_names,
    )
    if num_reps is None:
        num_reps = inferred.get("num_reps", None)
    if num_rounds_seq is None:
        num_rounds_seq = inferred.get("num_rounds_seq", 100)
    if num_rounds_batch is None:
        num_rounds_batch = inferred.get("num_rounds_batch", 30)
    if num_arms_seq is None:
        num_arms_seq = inferred.get("num_arms_seq", 1)
    if num_arms_batch is None:
        num_arms_batch = inferred.get("num_arms_batch", 50)

    _print_dataset_summary(
        results_path,
        exp_dir,
        problem=problem_seq,
        opt_names=opt_names_seq,
        num_arms=num_arms_seq,
        num_rounds=num_rounds_seq,
        num_reps=num_reps,
    )
    _print_dataset_summary(
        results_path,
        exp_dir,
        problem=problem_batch,
        opt_names=opt_names_batch,
        num_arms=num_arms_batch,
        num_rounds=num_rounds_batch,
        num_reps=num_reps,
    )

    suptitle1 = f"{problem_name}"
    print(suptitle1)
    fig_curves, axs_curves, seq_data, batch_data = plot_rl_comparison(
        results_path,
        exp_dir,
        opt_names_seq=opt_names_seq,
        opt_names_batch=opt_names_batch,
        problem_seq=problem_seq,
        problem_batch=problem_batch,
        num_reps=num_reps,
        num_rounds_seq=num_rounds_seq,
        num_rounds_batch=num_rounds_batch,
        num_arms_seq=num_arms_seq,
        num_arms_batch=num_arms_batch,
        suptitle=None,
        cum_dt_prop=problem in {"tlunar", "push"},
        opt_names_all=opt_names,
        show_titles=False,
        print_titles=True,
    )

    suptitle2 = f"{problem_name} Final Performance Comparison (±2 SE)"
    print(suptitle2)
    fig_final, axs_final, _, _ = plot_rl_final_comparison(
        results_path,
        exp_dir,
        opt_names_seq=opt_names_seq,
        opt_names_batch=opt_names_batch,
        problem_seq=problem_seq,
        problem_batch=problem_batch,
        num_reps=num_reps,
        num_rounds_seq=num_rounds_seq,
        num_rounds_batch=num_rounds_batch,
        num_arms_seq=num_arms_seq,
        num_arms_batch=num_arms_batch,
        suptitle=None,
        opt_names_all=opt_names,
        show_titles=False,
        print_titles=True,
    )

    return (fig_curves, axs_curves), (fig_final, axs_final), seq_data, batch_data


def compute_pareto_data(
    results_path: str,
    exp_dir: dict[str, str],
    opt_names: list[str],
    baseline_opt: str = "turbo-one",
    exclude_opts: list[str] | None = None,
    mode: str = "both",
):
    if exclude_opts is None:
        exclude_opts = ["random"]
    opt_names_filtered = [o for o in opt_names if o not in exclude_opts]

    all_returns = {}
    all_times = {}

    for problem, exp in exp_dir.items():
        problem_seq = problem
        problem_batch = f"{problem}:fn"

        inferred = _infer_params_from_configs(
            results_path,
            exp,
            problem_seq=problem_seq,
            problem_batch=problem_batch,
            opt_names=opt_names_filtered,
        )
        num_reps = inferred.get("num_reps", None)
        num_rounds_seq = inferred.get("num_rounds_seq", 100)
        num_rounds_batch = inferred.get("num_rounds_batch", 30)
        num_arms_seq = inferred.get("num_arms_seq", 1)
        num_arms_batch = inferred.get("num_arms_batch", 50)

        configs_to_load = []
        if mode in ("seq", "both"):
            configs_to_load.append(
                (f"{problem}_seq", problem_seq, num_arms_seq, num_rounds_seq)
            )
        if mode in ("batch", "both"):
            configs_to_load.append(
                (f"{problem}_batch", problem_batch, num_arms_batch, num_rounds_batch)
            )

        for label, prob, num_arms, num_rounds in configs_to_load:
            try:
                data_locator, traces = load_rl_traces(
                    results_path,
                    exp,
                    opt_names_filtered,
                    num_arms=num_arms,
                    num_rounds=num_rounds,
                    num_reps=num_reps,
                    problem=prob,
                )
                returns_by_opt = _mean_final_by_optimizer(data_locator, traces)
                all_returns[label] = returns_by_opt
            except ValueError:
                pass

            try:
                data_locator_dt, traces_dt = load_rl_traces(
                    results_path,
                    exp,
                    opt_names_filtered,
                    num_arms=num_arms,
                    num_rounds=num_rounds,
                    num_reps=num_reps,
                    problem=prob,
                    key="dt_prop",
                )
                traces_cum = _cum_dt_prop_from_dt_prop_traces(traces_dt)
                times_by_opt = _mean_final_by_optimizer(data_locator_dt, traces_cum)
                all_times[label] = times_by_opt
            except ValueError:
                pass

    problems = sorted(set(all_returns.keys()) & set(all_times.keys()))
    opts_in_data = set()
    for p in problems:
        opts_in_data.update(all_returns[p].keys())
        opts_in_data.update(all_times[p].keys())
    opts_in_data = [o for o in opt_names_filtered if o in opts_in_data]

    r_matrix = np.full((len(problems), len(opts_in_data)), np.nan)
    t_matrix = np.full((len(problems), len(opts_in_data)), np.nan)

    for i_p, p in enumerate(problems):
        for i_o, o in enumerate(opts_in_data):
            if p in all_returns and o in all_returns[p]:
                r_matrix[i_p, i_o] = all_returns[p][o]
            if p in all_times and o in all_times[p]:
                t_matrix[i_p, i_o] = all_times[p][o]

    if baseline_opt not in opts_in_data:
        raise ValueError(
            f"Baseline optimizer {baseline_opt!r} not found in data. Available: {opts_in_data}"
        )
    i_baseline = opts_in_data.index(baseline_opt)

    r_baseline = r_matrix[:, i_baseline : i_baseline + 1]
    r_centered = r_matrix - r_baseline
    rms = np.sqrt(np.nanmean(r_centered**2, axis=1, keepdims=True))
    rms = np.where(rms == 0, 1.0, rms)
    r_normalized = r_centered / rms

    t_baseline = t_matrix[:, i_baseline : i_baseline + 1]
    speedup = t_baseline / t_matrix

    return {
        "problems": problems,
        "opt_names": opts_in_data,
        "r_normalized": r_normalized,
        "speedup": speedup,
        "r_raw": r_matrix,
        "t_raw": t_matrix,
        "baseline_opt": baseline_opt,
    }
