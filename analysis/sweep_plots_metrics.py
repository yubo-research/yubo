"""Numeric helpers for sweep plot bar panels."""

import numpy as np


def _mean_ybest_mean_sem_per_curve(
    all_curves: list,
) -> tuple[list[float], list[float]]:
    means: list[float] = []
    sems: list[float] = []
    for curves in all_curves:
        n_reps, _ = curves.shape
        mean_per_rep = np.nanmean(curves, axis=1)
        means.append(float(np.nanmean(mean_per_rep)))
        sems.append(float(np.nanstd(mean_per_rep) / np.sqrt(n_reps)) if n_reps > 0 else 0.0)
    return means, sems


def _bar_heights_shift_if_negative(bar_means: list[float]) -> tuple[list[float], float]:
    """If any mean is negative, subtract min(bar_means) so the smallest bar is at zero."""
    shift = 0.0
    if bar_means:
        lo = min(bar_means)
        if lo < 0:
            shift = lo
    heights = [m - shift for m in bar_means]
    return heights, shift
