"""Publication-style matplotlib helpers for sweep plots."""

from pathlib import Path

import matplotlib.pyplot as plt

DEFAULT_SYNTH_10D_ENV_TAGS = (
    "f:sphere-10d",
    "f:ackley-10d",
    "f:booth-10d",
    "f:rosenbrock-10d",
)

FOUR_ENV_PANEL_LABELS = ("a", "b", "c", "d")


def _panel_label_text(letter: str) -> str:
    return f"({letter})"


def _scaled_fontsize(scale: float, *, minimum: float = 8.0) -> float:
    base = float(plt.rcParams.get("font.size", 12.0))
    return max(minimum, base * scale)


def _style_publication_axes(ax) -> None:
    spines = getattr(ax, "spines", None)
    if spines is not None:
        for spine_name in ("top", "right"):
            spine = spines.get(spine_name)
            if spine is not None:
                spine.set_visible(False)
    if hasattr(ax, "tick_params"):
        ax.tick_params(direction="out")
    if hasattr(ax, "set_axisbelow"):
        ax.set_axisbelow(True)


def _add_panel_caption_label(
    ax,
    letter: str,
    *,
    fontsize: float | None = None,
) -> None:
    label_fontsize = fontsize if fontsize is not None else _scaled_fontsize(0.72, minimum=11.0)
    ax.text(
        0.01,
        1.01,
        _panel_label_text(letter),
        transform=ax.transAxes,
        fontsize=label_fontsize,
        fontweight="semibold",
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.2},
    )


def _default_results_dir() -> Path:
    return Path.home() / "Projects/yubo/results"
