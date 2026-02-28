"""Types for plotting traces."""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from analysis.data_locator import DataLocator


class RLTracesWithCumDtProp(NamedTuple):
    data_locator: DataLocator
    traces: np.ndarray
    cum_dt_prop: Optional[dict[str, float]]


class PlotRLComparisonResult(NamedTuple):
    fig: object
    axs: object
    seq: RLTracesWithCumDtProp
    batch: Optional[RLTracesWithCumDtProp]


class PlotRLFinalComparisonResult(NamedTuple):
    fig: object
    axs: object
    seq: RLTracesWithCumDtProp
    batch: Optional[RLTracesWithCumDtProp]
