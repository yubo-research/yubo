import analysis.plotting as ap
from analysis.data_locator import DataLocator


class PlotCompare:
    def __init__(self, data_locator, optimizers, renames={}):
        self._data_locator = data_locator
        self._optimizers = optimizers
        self._renames = renames

    def __call__(self, ax, exp_name, num_arms, num_dim, i_agg=-1):
        self._plot_perf(ax, exp_name, num_arms, num_dim, i_agg, renames=self._renames)

    def _plot_perf(self, ax, exp_name, num_arms, num_dim, i_agg=-1, renames=None):
        ap.plot_sorted_agg(
            ax,
            self._data_locator,
            renames=renames,
            i_agg=i_agg,
        )
        if num_dim == 1:
            d = "dimension"
        else:
            d = "dimensions"
        ax.set_title(f"{num_arms} arms,  {num_dim} {d}", fontsize=14)
        ax.set_ylabel("$y_{max}$\n(normalized)")


def plot_compare(ax, data_locator, i_agg=-1, renames=None, b_sort=True, highlight=None, y_or_time="y"):
    ap.plot_sorted_agg(
        ax,
        data_locator,
        renames=renames,
        i_agg=i_agg,
        b_sort=b_sort,
        highlight=highlight,
    )

    if data_locator.num_dim is not None and data_locator.num_arms is not None:
        ax.set_title(f"num_dim = {data_locator.num_dim}  num_arms = {data_locator.num_arms}", fontsize=14)
        ax.set_ylim([-0.1, 1.03])

    if y_or_time == "y":
        ax.set_ylabel("score($y_{max}$)")
    else:
        ax.set_ylabel("score($t_{proposal}$)")


def pc_normal(
    results_path, exp_dir, ax, num_dim, num_arms, i_agg, opt_names, renames, num_reps=None, problems: set = None, highlight=None, y_or_time="y", b_sort=True
):
    plot_compare(
        ax,
        DataLocator(
            results_path,
            exp_dir,
            num_dim=num_dim,
            num_arms=num_arms,
            opt_names=opt_names,
            num_reps=num_reps,
            problems=problems,
            key="return" if y_or_time == "y" else "cum_dt_prop",
            grep_for="TRACE" if y_or_time == "y" else "ITER",
        ),
        renames=renames,
        i_agg=i_agg,
        highlight=highlight,
        y_or_time=y_or_time,
        b_sort=b_sort,
    )
