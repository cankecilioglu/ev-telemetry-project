"""Model diagnostics: comparison, error breakdown and a prediction window.

Kept separate from ``exploration.py`` because these figures need fitted models,
not just the cleaned table.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from . import config
from .plotting import COLORS, save_figure


def plot_model_comparison(metrics: pd.DataFrame, directory: Path | None = None) -> Path:
    """Horizontal bars of test MAE for every method (lower is better)."""
    ordered = metrics.sort_values("MAE_A", ascending=True)
    colours = [COLORS.get(key, "#333333") for key in ordered["key"]]

    fig, axis = plt.subplots(figsize=(8, 4))
    bars = axis.barh(ordered["method"], ordered["MAE_A"], color=colours)
    for bar, value in zip(bars, ordered["MAE_A"]):
        axis.text(value + 0.6, bar.get_y() + bar.get_height() / 2, f"{value:.1f} A", va="center")
    axis.set_xlabel("Test MAE [A]  (lower is better)")
    axis.set_title("Estimation error on the same held-out observations")
    axis.set_xlim(0, float(ordered["MAE_A"].max()) * 1.18)
    fig.tight_layout()
    return save_figure(fig, "fig05_model_comparison.png", directory)


def plot_error_analysis(
    error_speed: pd.DataFrame,
    error_regime: pd.DataFrame,
    directory: Path | None = None,
) -> Path:
    """Where do the models fail? MAE by speed band and by pedal regime."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    axis = axes[0]
    pivot = error_speed.pivot(index="speed_band", columns="method", values="MAE_A")
    for method in pivot.columns:
        axis.plot(
            range(len(pivot)),
            pivot[method],
            marker="o",
            label=method,
            color=COLORS.get(method, None),
        )
    axis.set_xticks(range(len(pivot)))
    axis.set_xticklabels([str(label) for label in pivot.index])
    axis.set_xlabel("Speed band [km/h]")
    axis.set_ylabel("MAE [A]")
    axis.set_title("Error grows with speed")
    axis.legend(fontsize=8)

    axis = axes[1]
    pivot = error_regime.pivot(index="regime", columns="method", values="MAE_A")
    width = 0.2
    for offset, method in enumerate(pivot.columns):
        x = [i + (offset - 1.5) * width for i in range(len(pivot))]
        axis.bar(x, pivot[method], width=width, label=method, color=COLORS.get(method, None))
    axis.set_xticks(range(len(pivot)))
    axis.set_xticklabels(list(pivot.index))
    axis.set_ylabel("MAE [A]")
    axis.set_title("Error by operating regime")
    axis.set_ylim(0, float(error_regime["MAE_A"].max()) * 1.55)
    axis.legend(fontsize=8, ncol=2)

    fig.suptitle("Residual analysis on the test set", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save_figure(fig, "fig06_error_analysis.png", directory)


def plot_prediction_window(
    predictions: pd.DataFrame,
    trip: int | None = None,
    minutes: float = 20.0,
    directory: Path | None = None,
) -> Path:
    """Measured vs estimated current over a window of the test set."""
    subset = predictions if trip is None else predictions.loc[predictions[config.GROUP_COL] == trip]
    if subset.empty:
        raise ValueError(f"No test predictions for trip {trip}")

    subset = subset.sort_values(config.TIME_COL)
    start = subset[config.TIME_COL].min()
    window = subset.loc[subset[config.TIME_COL] <= start + pd.Timedelta(minutes=minutes)]

    fig, axis = plt.subplots(figsize=(11, 4))
    axis.plot(window[config.TIME_COL], window["y_true"], color="black", linewidth=1.0, label="measured")
    axis.plot(window[config.TIME_COL], window["rf"], color=COLORS["rf"], linewidth=0.9, label="random forest")
    axis.plot(
        window[config.TIME_COL],
        window["physics"],
        color=COLORS["physics"],
        linewidth=0.9,
        label="physics calibration",
    )
    axis.axhline(0, color="grey", linewidth=0.6)
    axis.set_xlabel("Time [UTC]")
    axis.set_ylabel("Battery current [A]")
    axis.set_title(f"Test window - trip {int(subset[config.GROUP_COL].iloc[0])}, first {minutes:.0f} min")
    # Headroom so the legend does not sit on top of the measured spikes.
    top = max(window["y_true"].max(), window["rf"].max(), window["physics"].max())
    axis.set_ylim(min(-20.0, window["y_true"].min() - 15), top * 1.12)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=9)
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout()
    return save_figure(fig, "fig07_prediction_window.png", directory)
