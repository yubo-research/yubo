"""Types for plotting module."""

from typing import NamedTuple

import numpy as np

from analysis.data_locator import DataLocator

# Re-export from plotting_trace_types
from .plotting_trace_types import (  # noqa: F401
    PlotRLComparisonResult,
    PlotRLFinalComparisonResult,
    RLTracesWithCumDtProp,
)


class PlotRLExperimentResult(NamedTuple):
    fig: object
    ax: object
    data_locator: DataLocator
    traces: np.ndarray


class PlotRLExperimentVsTimeResult(NamedTuple):
    fig: object
    ax: object
    data_locator: DataLocator
    traces: np.ndarray
    t: np.ndarray


class PlotResultsResult(NamedTuple):
    curves: tuple[object, object]
    final: tuple[object, object]
    seq_data: object
    batch_data: object
