import matplotlib.pyplot as plt

from .plotting_3_axes import _plot_vs_fe, _plot_vs_time
from .plotting_3_data import (
    _infer_params_from_configs,
    _load_r_and_dt,
    _mean_final_cum_dt_total_by_opt,
)


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

    dl_seq_r, tr_seq_r, tr_seq_dt_prop, tr_seq_dt_total = _load_r_and_dt(
        results_path,
        exp_dir,
        opt_names=opt_names,
        num_arms=1,
        num_rounds=num_rounds_seq,
        num_reps=num_reps,
        problem=problem_seq,
    )
    dl_batch_r, tr_batch_r, tr_batch_dt_prop, tr_batch_dt_total = _load_r_and_dt(
        results_path,
        exp_dir,
        opt_names=opt_names,
        num_arms=num_arms_batch,
        num_rounds=num_rounds_batch,
        num_reps=num_reps,
        problem=problem_batch,
    )

    # Legend timing (mean final sum(dt_prop) over reps)
    seq_opts = dl_seq_r.optimizers()
    batch_opts = dl_batch_r.optimizers()
    seq_t_final = _mean_final_cum_dt_total_by_opt(tr_seq_dt_prop, seq_opts)
    batch_t_final = _mean_final_cum_dt_total_by_opt(tr_batch_dt_prop, batch_opts)

    fig, axs = plt.subplots(2, 2, figsize=figsize, sharey=False)

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
