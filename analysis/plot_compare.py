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
            exp_name,
            self._optimizers,
            renames=renames,
            i_agg=i_agg,
            old_way=False,
        )
        if num_dim == 1:
            d = "dimension"
        else:
            d = "dimensions"
        ax.set_title(f"{num_arms} arms,  {num_dim} {d}", fontsize=14)
        ax.set_ylabel("$y_{max}$\n(normalized)")


def plot_compare(ax, data_locator, i_agg=-1, renames=None, b_sort=True):
    ap.plot_sorted_agg(
        ax,
        data_locator,
        renames=renames,
        i_agg=i_agg,
        b_sort=b_sort,
    )

    if data_locator.num_dim is not None and data_locator.num_arms is not None:
        ax.set_title(f"num_dim = {data_locator.num_dim}  num_arms = {data_locator.num_arms}", fontsize=14)
        ax.set_ylim([0, 1.03])
    ax.set_ylabel("score")


def pc_normal(results_path, exp_dir, ax, num_dim, num_arms, i_agg, opt_names, renames, num_reps=None):
    plot_compare(
        ax,
        DataLocator(
            results_path,
            exp_dir,
            num_dim=num_dim,
            num_arms=num_arms,
            opt_names=opt_names,
            num_reps=num_reps,
        ),
        renames=renames,
        i_agg=i_agg,
    )
