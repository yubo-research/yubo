"""Single combined result type (split to satisfy concrete_types_per_file)."""

from typing import NamedTuple


class PlotResultsCombinedResult(NamedTuple):
    fig: object
    axs: object
    seq_data: object
    batch_data: object
