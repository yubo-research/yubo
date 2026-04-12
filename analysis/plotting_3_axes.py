import numpy as np

import analysis.plotting as ap

from .plotting_3_data import _best_so_far


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
    xlim_opt_name: str = "turbo-enn-fit/acq_type=ucb",
):
    opts = dl_r.optimizers()
    z_r = tr_r.squeeze(0)
    z_dt = tr_dt.squeeze(0)

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
