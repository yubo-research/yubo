"""Generate the failure-clock event schedule for the TuRBO note."""

from __future__ import annotations

import os
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/yubo-matplotlib-cache")

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).with_suffix(".pdf")
OUT_PNG = Path(__file__).with_suffix(".png")


def _bracket(ax: plt.Axes, x0: float, x1: float, y: float, label: str) -> None:
    ax.plot([x0, x1], [y, y], color="0.25", linewidth=0.8)
    ax.plot([x0, x0], [y - 0.04, y + 0.04], color="0.25", linewidth=0.8)
    ax.plot([x1, x1], [y - 0.04, y + 0.04], color="0.25", linewidth=0.8)
    ax.text((x0 + x1) / 2.0, y - 0.09, label, ha="center", va="top", fontsize=8.5, color="0.2")


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 2.25), constrained_layout=True)

    ax.hlines(0.0, 0.0, 8.0, color="black", linewidth=1.1)

    for k in range(1, 7):
        ax.vlines(k, -0.08, 0.40, color="black", linewidth=1.2)
        ax.text(k, 0.48, f"{k}", ha="center", va="bottom", fontsize=9)

    ax.vlines(7, -0.10, 0.62, color="black", linewidth=2.0)
    ax.text(7, 0.70, "7", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.text(1.0, 0.82, "first shrink", ha="center", va="bottom", fontsize=9)
    ax.text(7.0, 0.82, "default restart\nthreshold", ha="center", va="bottom", fontsize=9)
    ax.text(4.0, 0.62, "successive failure-driven contractions", ha="center", va="bottom", fontsize=9)

    _bracket(ax, 0.0, 1.0, -0.36, "no shrink")
    _bracket(ax, 1.0, 7.0, -0.58, r"$\lfloor T/D\rfloor$ shrink-event bound")
    _bracket(ax, 7.0, 8.0, -0.36, "restart possible")

    ax.set_xlim(-0.05, 8.05)
    ax.set_ylim(-0.88, 1.18)
    ax.set_xticks(np.arange(0, 9, 1))
    ax.set_xlabel(r"budget multiple $T/D$")
    ax.set_yticks([])
    ax.tick_params(axis="x", length=3.5, width=0.8, color="0.2")

    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("0.2")
    ax.spines["bottom"].set_linewidth(0.8)

    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=260, bbox_inches="tight")
    print(OUT)
    print(OUT_PNG)


if __name__ == "__main__":
    main()
