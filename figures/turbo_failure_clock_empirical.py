"""Empirical all-failure TuRBO trust-region clock trace."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/yubo-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from enn.turbo.config.turbo_tr_config import TurboTRConfig
from enn.turbo.turbo_trust_region import TurboTrustRegion


OUT = Path(__file__).with_suffix(".pdf")
OUT_PNG = Path(__file__).with_suffix(".png")
DIMS = (10_000, 500_000, 10_000_000, 1_000_000_000)
NUM_ARMS = 1
NUM_SHRINKS = 7


def _dim_label(num_dim: int) -> str:
    if num_dim == 10_000:
        return r"$D=10^4$"
    if num_dim == 500_000:
        return r"$D=5{\times}10^5$"
    if num_dim == 10_000_000:
        return r"$D=10^7$"
    if num_dim == 1_000_000_000:
        return r"$D=10^9$"
    return rf"$D={num_dim:g}$"


def failure_trace(num_dim: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Return event-level trace and first restart time.

    The counter is advanced at event level so the diagnostic remains finite for
    D=1e9. The length update and restart predicate are the actual trust-region
    state transitions.
    """
    tr = TurboTrustRegion(TurboTRConfig(), num_dim)
    tr.validate_request(NUM_ARMS)
    if tr.failure_tolerance != num_dim:
        raise AssertionError((num_dim, tr.failure_tolerance))

    length0 = float(tr.length)
    x = [1_000.0]
    y = [1.0]
    assigned = 0

    for _ in range(NUM_SHRINKS):
        tr.failure_counter = tr.failure_tolerance - 1
        before = float(tr.length)
        tr._update_counters_and_length(improved=False)
        if not np.isclose(tr.length, 0.5 * before):
            raise AssertionError((num_dim, before, tr.length))
        assigned += tr.failure_tolerance
        x.append(assigned)
        y.append(float(tr.length) / length0)

    if not tr.needs_restart():
        raise AssertionError((num_dim, tr.length, tr.length_min))
    restart_at = float(assigned)
    tr.restart()
    if not np.isclose(tr.length / length0, 1.0):
        raise AssertionError((num_dim, tr.length, length0))

    return np.asarray(x, dtype=float), np.asarray(y, dtype=float), restart_at


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)

    colors = ["#000000", "#4a4a4a", "#8a4f00", "#1f5a85"]
    linestyles = ["-", "--", "-.", ":"]

    for color, linestyle, num_dim in zip(colors, linestyles, DIMS):
        x, y, restart_at = failure_trace(num_dim)
        ax.step(
            x,
            y,
            where="post",
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=_dim_label(num_dim),
        )
        ax.plot(x[1:], y[1:], marker="o", linestyle="none", color=color, markersize=3.5)
        ax.vlines(restart_at, y[-1], 1.0, color=color, linestyle=linestyle, linewidth=1.5)
        ax.plot([restart_at], [1.0], marker="^", linestyle="none", color=color, markersize=5.0)

    ax.set_xscale("log")
    ax.set_yscale("log", base=2)
    ax.set_xlim(1_000, 10_000_000_000)
    ax.set_ylim(2 ** -7.35, 1.18)

    y_ticks = [2.0 ** (-k) for k in range(NUM_SHRINKS + 1)]
    y_labels = [r"$1$"] + [rf"$2^{{-{k}}}$" for k in range(1, NUM_SHRINKS + 1)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    ax.set_xlabel(r"assigned evaluations $T$")
    ax.set_ylabel(r"normalized trust-region length $\Delta_T/\Delta_0$")
    ax.plot([], [], marker="^", linestyle="none", color="0.25", markersize=5.0, label="restart reset")

    ax.grid(True, which="major", color="0.86", linewidth=0.75)
    ax.grid(True, which="minor", axis="x", color="0.92", linewidth=0.45)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=260, bbox_inches="tight")
    print(OUT)
    print(OUT_PNG)


if __name__ == "__main__":
    main()
