"""Generate the normalized failure-clock plot for the TuRBO note."""

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


def main() -> None:
    c = np.arange(0, 12, dtype=float)
    n_shrink = np.floor(c)

    fig, ax = plt.subplots(figsize=(4.7, 3.2), constrained_layout=True)

    ax.step(
        c,
        n_shrink,
        where="post",
        color="black",
        linewidth=1.8,
    )
    ax.axvline(1.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.axvline(7.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.text(1.08, 0.35, r"$T=D$", fontsize=9, color="0.25")
    ax.text(7.08, 0.35, r"$T=7D$", fontsize=9, color="0.25")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_yticks(np.arange(0, 11, 1))
    ax.set_xlabel(r"budget multiple $c=T/D$")
    ax.set_ylabel(r"upper bound on shrink events")
    ax.set_title(r"$N_{\downarrow}(T)\leq\lfloor T/D\rfloor$", loc="left", fontsize=11)

    ax.grid(True, color="0.88", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=240, bbox_inches="tight")
    print(OUT)
    print(OUT_PNG)


if __name__ == "__main__":
    main()
