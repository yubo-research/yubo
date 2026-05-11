"""Shared bottom legend row for 2x2 sweep figures."""

from analysis.sweep_plots_style import _scaled_fontsize


def _shared_bottom_legend_kwargs(
    num_labels: int,
) -> dict[str, float | bool | int | str]:
    return {
        "loc": "center",
        "ncols": min(num_labels, 5),
        "frameon": False,
        "fontsize": _scaled_fontsize(0.74, minimum=10.0),
        "handlelength": 1.7,
        "columnspacing": 1.0,
        "handletextpad": 0.45,
        "borderaxespad": 0.0,
    }


def _split_bottom_legend_kwargs(
    num_labels: int,
    title: str,
) -> dict[str, float | bool | int | str]:
    fontsize = _scaled_fontsize(0.68, minimum=9.0)
    return {
        "loc": "center",
        "ncols": min(num_labels, 3),
        "title": title,
        "frameon": False,
        "fontsize": fontsize,
        "title_fontsize": fontsize,
        "handlelength": 1.6,
        "columnspacing": 0.9,
        "handletextpad": 0.4,
        "labelspacing": 0.35,
        "borderaxespad": 0.0,
    }


def _render_matching_column_legends(
    ax_legend,
    left_handles: list,
    left_labels: list[str],
) -> None:
    ax_legend.legend(
        left_handles,
        left_labels,
        **_shared_bottom_legend_kwargs(len(left_labels)),
    )


def _render_split_column_legends(
    fig,
    gs,
    left_handles: list,
    left_labels: list[str],
    left_param: str,
    right_handles: list,
    right_labels: list[str],
    right_param: str,
) -> None:
    ax_left_legend = fig.add_subplot(gs[4, 0])
    ax_right_legend = fig.add_subplot(gs[4, 1])
    ax_left_legend.set_axis_off()
    ax_right_legend.set_axis_off()
    ax_left_legend.legend(
        left_handles,
        left_labels,
        **_split_bottom_legend_kwargs(num_labels=len(left_labels), title=left_param),
    )
    ax_right_legend.legend(
        right_handles,
        right_labels,
        **_split_bottom_legend_kwargs(num_labels=len(right_labels), title=right_param),
    )


def _render_bottom_legend_row(
    fig,
    gs,
    column_legends: dict[int, tuple[list, list[str], str]],
    legend_handles,
    legend_labels,
) -> None:
    if len(column_legends) == 2:
        ax_legend = fig.add_subplot(gs[4, :])
        ax_legend.set_axis_off()
        left_handles, left_labels, left_param = column_legends[0]
        right_handles, right_labels, right_param = column_legends[1]
        if left_labels == right_labels:
            _render_matching_column_legends(ax_legend, left_handles, left_labels)
            return
        _render_split_column_legends(
            fig,
            gs,
            left_handles,
            left_labels,
            left_param,
            right_handles,
            right_labels,
            right_param,
        )
        return
    if legend_handles and legend_labels:
        ax_legend = fig.add_subplot(gs[4, :])
        ax_legend.set_axis_off()
        ax_legend.legend(
            legend_handles,
            legend_labels,
            **_shared_bottom_legend_kwargs(len(legend_labels)),
        )
