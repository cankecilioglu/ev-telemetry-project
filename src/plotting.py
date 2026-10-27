"""Plot styling and saving helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never shown

import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402

COLORS = {
    "speed": "#1f4e79",
    "current": "#c0392b",
    "pedal": "#2e8b57",
    "brake": "#8e44ad",
    "voltage": "#b8860b",
    "soc": "#16a085",
    "eco": "#e67e22",
    "baseline": "#7f8c8d",
    "linear": "#8e44ad",
    "physics": "#1f4e79",
    "rf": "#c0392b",
    "grid": "#d9d9d9",
}


def apply_style() -> None:
    """Set a consistent, print-friendly style for every figure."""
    plt.rcParams.update(
        {
            "figure.dpi": config.FIGURE_DPI,
            "savefig.dpi": config.FIGURE_DPI,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.autolayout": False,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, filename: str, directory: Path | None = None) -> Path:
    """Save ``fig`` under ``results/figures`` and return the path."""
    directory = Path(directory or config.FIGURE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
