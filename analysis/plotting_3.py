import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import analysis.data_sets as ds
import analysis.plotting as ap
from analysis.data_locator import DataLocator


def _normalize_results_and_exp_dir(results_path: str, exp_dir: str) -> tuple[str, str]:
    """Normalize (results_path, exp_dir) to what DataLocator expects."""
    rp = Path(results_path).expanduser()
    ed = Path(exp_dir).expanduser()

    # results_path is repo root and has results/<exp_dir>
    if (rp / "results" / exp_dir).exists():
        return str(rp / "results"), exp_dir

    # exp_dir is "results/<exp>" while results_path ends with ".../results"
    if rp.name == "results" and exp_dir.startswith("results/"):
        exp_dir = exp_dir[len("results/") :]
        return str(rp), exp_dir

    # exp_dir is a path to the experiment directory
    if ed.exists() and ed.is_dir():
        return str(ed.parent), ed.name

    return str(rp), exp_dir


def _infer_params_from_configs(
    results_path: str,
    exp_dir: str,
    *,
    problem_seq: str,
    problem_batch: str,
    opt_names: list[str],
) -> dict[str, int]:
    """Infer (num_rounds_seq, num_rounds_batch, num_arms_batch, num_reps) from config.json."""
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
    na_batch = _uniq_int([r["num_arms"] for r in batch])
    reps_seq = _uniq_int([r["num_reps"] for r in seq])
    reps_batch = _uniq_int([r["num_reps"] for r in batch])

    if nr_seq is not None:
        out["num_rounds_seq"] = nr_seq
    if nr_batch is not None:
        out["num_rounds_batch"] = nr_batch
    if na_batch is not None:
        out["num_arms_batch"] = na_batch

    reps = reps_seq if reps_seq is not None else reps_batch
    if reps is not None:
        out["num_reps"] = reps

    return out


def _load_traces(
    results_path: str,
    exp_dir: str,
    *,
    opt_names: list[str],
    num_arms: int,
    num_rounds: int,
    num_reps: int,
    problem: str,
    key: str,
) -> tuple[DataLocator, np.ndarray]:
    results_path, exp_dir = _normalize_results_and_exp_dir(results_path, exp_dir)
    dl = DataLocator(
        results_path,
        exp_dir,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
        opt_names=opt_names,
        problems={problem},
        key=key,
    )
    traces = ds.load_multiple_traces(dl)
    return dl, traces


def _best_so_far(ys: np.ndarray) -> np.ndarray:
    # numpy.ma does not provide maximum.accumulate; handle masks explicitly.
    if np.ma.isMaskedArray(ys):
        data = ys.filled(-np.inf)
        acc = np.maximum.accumulate(data, axis=-1)
        acc[~np.isfinite(acc)] = np.nan
        return np.ma.array(acc, mask=np.ma.getmaskarray(ys))
    return np.maximum.accumulate(ys, axis=-1)


def _sum_traces(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise sum preserving masks when present."""
    if np.ma.isMaskedArray(a) or np.ma.isMaskedArray(b):
        return np.ma.array(a) + np.ma.array(b)
    return a + b


def _mean_final_cum_dt_total_by_opt(traces_dt_total: np.ndarray, optimizers: list[str]) -> dict[str, float]:
    z = traces_dt_total.squeeze(0)  # [n_opt, n_rep, n_round]
    out: dict[str, float] = {}
    for i_opt, opt_name in enumerate(optimizers):
        dt = z[i_opt, ...]
        cum = np.ma.cumsum(dt, axis=-1) if np.ma.isMaskedArray(dt) else np.cumsum(dt, axis=-1)
        out[opt_name] = float(np.ma.mean(cum[:, -1]))
    return out


def plot_results_grid(
    results_path: str,
    exp_dir: str,
    *,
    opt_names: list[str],
    problem: str,
    num_reps: int | None = None,
    num_rounds_seq: int | None = None,
    num_rounds_batch: int | None = None,
    num_arms_batch: int | None = None,
    figsize: tuple[int, int] = (14, 10),
):
    """Plot a 2x2 grid:

    (1,1) sequential: best-so-far return vs # function evals
    (2,1) batch:      best-so-far return vs # function evals
    (1,2) sequential: y_max vs cumsum(dt_prop + dt_eval)
    (2,2) batch:      y_max vs cumsum(dt_prop + dt_eval)

    Returns (fig, axs) where axs is a 2x2 array.
    """

    problem_seq = problem
    problem_batch = f"{problem}:fn"

    inferred = _infer_params_from_configs(
        results_path,
        exp_dir,
        problem_seq=problem_seq,
        problem_batch=problem_batch,
        opt_names=opt_names,
    )

    if num_reps is None:
        num_reps = inferred.get("num_reps", 30)
    if num_rounds_seq is None:
        num_rounds_seq = inferred.get("num_rounds_seq", 100)
    if num_rounds_batch is None:
        num_rounds_batch = inferred.get("num_rounds_batch", 30)
    if num_arms_batch is None:
        num_arms_batch = inferred.get("num_arms_batch", 50)

    # Load return + (dt_prop + dt_eval) for sequential and batch
    dl_seq_r, tr_seq_r = _load_traces(
        results_path,
        exp_dir,
        opt_names=opt_names,
        num_arms=1,
        num_rounds=num_rounds_seq,
        num_reps=num_reps,
        problem=problem_seq,
        key="rreturn",
    )
    dl_seq_dt_prop, tr_seq_dt_prop = _load_traces(
        results_path,
        exp_dir,
        opt_names=opt_names,
        num_arms=1,
        num_rounds=num_rounds_seq,
        num_reps=num_reps,
        problem=problem_seq,
        key="dt_prop",
    )
    _, tr_seq_dt_eval = _load_traces(
        results_path,
        exp_dir,
        opt_names=opt_names,
        num_arms=1,
        num_rounds=num_rounds_seq,
        num_reps=num_reps,
        problem=problem_seq,
        key="dt_eval",
    )
    tr_seq_dt_total = _sum_traces(tr_seq_dt_prop, tr_seq_dt_eval)

    dl_batch_r, tr_batch_r = _load_traces(
        results_path,
        exp_dir,
        opt_names=opt_names,
        num_arms=num_arms_batch,
        num_rounds=num_rounds_batch,
        num_reps=num_reps,
        problem=problem_batch,
        key="rreturn",
    )
    dl_batch_dt_prop, tr_batch_dt_prop = _load_traces(
        results_path,
        exp_dir,
        opt_names=opt_names,
        num_arms=num_arms_batch,
        num_rounds=num_rounds_batch,
        num_reps=num_reps,
        problem=problem_batch,
        key="dt_prop",
    )
    _, tr_batch_dt_eval = _load_traces(
        results_path,
        exp_dir,
        opt_names=opt_names,
        num_arms=num_arms_batch,
        num_rounds=num_rounds_batch,
        num_reps=num_reps,
        problem=problem_batch,
        key="dt_eval",
    )
    tr_batch_dt_total = _sum_traces(tr_batch_dt_prop, tr_batch_dt_eval)

    # Legend timing (mean final sum(dt_prop) over reps)
    seq_opts = dl_seq_r.optimizers()
    batch_opts = dl_batch_r.optimizers()
    seq_t_final = _mean_final_cum_dt_total_by_opt(tr_seq_dt_prop, seq_opts)
    batch_t_final = _mean_final_cum_dt_total_by_opt(tr_batch_dt_prop, batch_opts)

    fig, axs = plt.subplots(2, 2, figsize=figsize, sharey=False)

    def _plot_vs_fe(ax, dl, tr_r, *, num_arms: int, title: str, t_final: dict[str, float]):
        opts = dl.optimizers()
        z = tr_r.squeeze(0)
        for i_opt, opt_name in enumerate(opts):
            y = _best_so_far(z[i_opt, ...])
            x = num_arms * (1 + np.arange(y.shape[1]))
            label = opt_name
            if opt_name in t_final:
                label = f"{opt_name} ({t_final[opt_name]:.1f}s)"
            ap.filled_err(
                x=x,
                ys=y,
                ax=ax,
                se=True,
                alpha=0.25,
                color=ap.colors[i_opt],
                color_line=ap.colors[i_opt],
                label=label,
                marker=ap.markers[i_opt],
                max_markers=10,
                markersize=5,
            )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("# Function Evaluations", fontsize=11)
        ax.set_ylabel("Return (best so far)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)

    def _plot_vs_time(
        ax,
        dl_r,
        tr_r,
        tr_dt,
        *,
        title: str,
        t_final: dict[str, float],
        xlim_opt_name: str = "turbo-enn-fit-ucb",
    ):
        opts = dl_r.optimizers()
        z_r = tr_r.squeeze(0)
        z_dt = tr_dt.squeeze(0)

        # Set the time horizon to the reference optimizer's time range, if present.
        x_max = None
        if xlim_opt_name in opts:
            i_ref = opts.index(xlim_opt_name)
            dt_ref = z_dt[i_ref, ...]
            x_ref_rep = np.ma.cumsum(dt_ref, axis=-1) if np.ma.isMaskedArray(dt_ref) else np.cumsum(dt_ref, axis=-1)
            x_ref = np.ma.mean(x_ref_rep, axis=0) if np.ma.isMaskedArray(x_ref_rep) else x_ref_rep.mean(axis=0)
            x_max = float(x_ref[-1])

        for i_opt, opt_name in enumerate(opts):
            y = _best_so_far(z_r[i_opt, ...])
            dt = z_dt[i_opt, ...]
            x_rep = np.ma.cumsum(dt, axis=-1) if np.ma.isMaskedArray(dt) else np.cumsum(dt, axis=-1)
            x = np.ma.mean(x_rep, axis=0) if np.ma.isMaskedArray(x_rep) else x_rep.mean(axis=0)

            if x_max is not None:
                # Truncate to the reference optimizer's time horizon.
                keep = np.asarray(x) <= x_max
                n_keep = int(np.sum(keep))
                if n_keep <= 0:
                    continue
                x = x[:n_keep]
                y = y[:, :n_keep]

            label = opt_name
            if opt_name in t_final:
                label = f"{opt_name} ({t_final[opt_name]:.1f}s)"
            ap.filled_err(
                x=x,
                ys=y,
                ax=ax,
                se=True,
                alpha=0.25,
                color=ap.colors[i_opt],
                color_line=ap.colors[i_opt],
                label=label,
                marker=ap.markers[i_opt],
                max_markers=10,
                markersize=5,
            )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("cumsum(dt_prop + dt_eval) [s]", fontsize=11)
        ax.set_ylabel("Return (best so far)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
        if x_max is not None:
            ax.set_xlim(0, x_max)

    _plot_vs_fe(
        axs[0, 0],
        dl_seq_r,
        tr_seq_r,
        num_arms=1,
        title="Sequential",
        t_final=seq_t_final,
    )
    _plot_vs_fe(
        axs[1, 0],
        dl_batch_r,
        tr_batch_r,
        num_arms=num_arms_batch,
        title=f"Batch (num_arms / round = {num_arms_batch})",
        t_final=batch_t_final,
    )

    _plot_vs_time(
        axs[0, 1],
        dl_seq_r,
        tr_seq_r,
        tr_seq_dt_total,
        title="Sequential: y_max vs cumsum(dt_prop + dt_eval)",
        t_final=seq_t_final,
    )
    _plot_vs_time(
        axs[1, 1],
        dl_batch_r,
        tr_batch_r,
        tr_batch_dt_total,
        title="Batch: y_max vs cumsum(dt_prop + dt_eval)",
        t_final=batch_t_final,
    )

    fig.suptitle(f"{problem} results", fontsize=14, y=1.02)
    plt.tight_layout()
    return fig, axs
