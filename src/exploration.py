"""Exploratory figures.

Figure conventions used throughout the project
----------------------------------------------
* Every axis states its quantity **and** unit.
* Time-series lines are broken at sampling gaps; gaps are never bridged with a
  straight line, because that would draw driving that was never recorded.
* Each plot is saved to ``results/figures`` instead of being shown.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .plotting import COLORS, save_figure


def insert_gap_markers(frame: pd.DataFrame, max_gap_s: float = config.MAX_GAP_SECONDS) -> pd.DataFrame:
    """Insert one ``NaN`` row inside every real sampling gap.

    The line is then broken exactly where data are missing, while all measured
    samples keep their original timestamps (no resampling, no interpolation).
    """
    ordered = frame.sort_values(config.TIME_COL).copy()
    deltas = ordered[config.TIME_COL].diff().dt.total_seconds()
    gap_rows = []

    for index, gap_seconds in deltas[deltas > max_gap_s].items():
        marker = pd.Series(np.nan, index=ordered.columns, dtype=object)
        marker[config.TIME_COL] = ordered.at[index, config.TIME_COL] - pd.Timedelta(
            seconds=float(gap_seconds) / 2.0
        )
        marker[config.GROUP_COL] = ordered.at[index, config.GROUP_COL]
        gap_rows.append(marker)

    if gap_rows:
        ordered = pd.concat([ordered, pd.DataFrame(gap_rows)], ignore_index=True)
        ordered = ordered.sort_values(config.TIME_COL, kind="mergesort")
    return ordered


def plot_trip_timeseries(
    frame: pd.DataFrame,
    trip: int,
    filename: str | None = None,
    directory: Path | None = None,
) -> Path:
    """Four-panel time-series profile of one trip (speed, current, pedals, SOC)."""
    subset = frame.loc[frame[config.GROUP_COL] == trip]
    if subset.empty:
        raise ValueError(f"No rows for trip {trip}")

    grid = insert_gap_markers(subset, config.MAX_GAP_SECONDS)
    start = subset[config.TIME_COL].min()
    title_date = start.strftime("%Y-%m-%d")

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(grid[config.TIME_COL], grid["SPD"], color=COLORS["speed"], linewidth=0.9)
    axes[0].set_ylabel("Vehicle speed\n[km/h]")
    axes[0].set_title(f"Trip {trip} ({title_date}, UTC) - measured telemetry")

    axes[1].plot(grid[config.TIME_COL], grid[config.TARGET], color=COLORS["current"], linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_ylabel("Battery current\n[A]  (signed)")

    axes[2].plot(grid[config.TIME_COL], grid["ACC"], color=COLORS["pedal"], linewidth=0.8, label="ACC (accelerator)")
    axes[2].plot(grid[config.TIME_COL], grid["BRK"], color=COLORS["brake"], linewidth=0.8, label="BRK (brake)")
    axes[2].set_ylabel("Pedal travel\n[%]")
    axes[2].legend(loc="upper right", ncol=2)

    axes[3].plot(grid[config.TIME_COL], grid["CH"], color=COLORS["soc"], linewidth=0.9)
    axes[3].set_ylabel("State of charge\n[%]")
    axes[3].set_xlabel("Time [UTC]")

    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[3].xaxis.set_major_locator(mdates.AutoDateLocator())

    all_times = subset[config.TIME_COL].sort_values()
    deltas = all_times.diff().dt.total_seconds().dropna()
    n_gaps = int((deltas > config.MAX_GAP_SECONDS).sum())
    fig.text(
        0.01,
        0.995,
        f"Gaps in the line = missing samples ({n_gaps} gap(s) > {config.MAX_GAP_SECONDS:.0f} s). "
        "No interpolation applied.",
        fontsize=8,
        va="top",
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    name = filename or f"fig01_timeseries_cond{trip:02d}.png"
    return save_figure(fig, name, directory)


def plot_voltage_channel(frame: pd.DataFrame, directory: Path | None = None) -> Path:
    """Small multiples of pack voltage per trip, to justify the ``vol_valid`` flag."""
    trips = sorted(frame[config.GROUP_COL].unique())
    fig, axes = plt.subplots(3, 3, figsize=(11, 7), sharey=True)
    axes = axes.ravel()

    for axis, trip in zip(axes, trips):
        subset = frame.loc[frame[config.GROUP_COL] == trip]
        elapsed = (subset[config.TIME_COL] - subset[config.TIME_COL].min()).dt.total_seconds() / 60
        axis.plot(elapsed, subset["VOL"], color=COLORS["voltage"], linewidth=0.7)
        axis.axhspan(*config.VALID_RANGES["VOL"], color=COLORS["grid"], alpha=0.45, zorder=0)
        axis.set_title(f"Trip {trip}", fontsize=9)
        axis.set_xlabel("Elapsed time [min]", fontsize=8)
        axis.tick_params(labelsize=8)

    for axis in axes[len(trips):]:
        axis.set_visible(False)

    axes[0].set_ylabel("Pack voltage [V]")
    axes[3].set_ylabel("Pack voltage [V]")
    axes[6].set_ylabel("Pack voltage [V]")
    axes[6].set_ylim(-20, 430)
    fig.suptitle(
        "Pack-voltage channel: shaded band = physically plausible range (250-420 V)\n"
        "Only trips 7 and 8 carry a usable voltage signal",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return save_figure(fig, "fig02_voltage_channel_quality.png", directory)


def plot_pedal_response(frame: pd.DataFrame, directory: Path | None = None) -> Path:
    """Battery current as a function of pedal travel (2-D density, log counts).

    A hexbin is used instead of a scatter plot because 24 000 points would
    overplot; the colour shows how many samples fall in each cell.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

    specs = [
        (axes[0], "ACC", "Accelerator pedal travel [%]"),
        (axes[1], "BRK", "Brake pedal travel [%]"),
    ]
    for axis, column, label in specs:
        density = axis.hexbin(
            frame[column],
            frame[config.TARGET],
            gridsize=45,
            cmap="viridis",
            bins="log",
            mincnt=1,
        )
        correlation = frame[column].corr(frame[config.TARGET])
        axis.axhline(0, color="black", linewidth=0.6)
        axis.set_xlabel(label)
        axis.set_title(f"{column} vs current  (r = {correlation:+.2f})")
        fig.colorbar(density, ax=axis, label="samples per cell (log)")

    axes[0].set_ylabel("Battery current [A]  (signed)")
    fig.suptitle("Pedal inputs and battery current - all 9 trips", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, "fig03_pedal_response.png", directory)


def matched_eco_summary(
    frame: pd.DataFrame,
    trip: int = 4,
) -> tuple[pd.DataFrame, float]:
    """ECO effect per speed band, comparing only matched speed and pedal.

    Samples are grouped into speed bands and accelerator-pedal quintiles; a
    cell contributes only if it contains both ECO states. The per-band value
    is the weighted mean of the within-cell differences (ECO on - ECO off),
    weighted by the smaller of the two sample counts. The second return value
    is the overall weighted mean, in ampere.
    """
    subset = frame.loc[frame[config.GROUP_COL] == trip].sort_values(config.TIME_COL)
    if subset.empty:
        raise ValueError(f"No rows for trip {trip}")
    if subset["ECO_bin"].nunique(dropna=True) < 2:
        raise ValueError(f"Trip {trip} never switches ECO mode; nothing to compare")

    moving = subset.loc[subset["SPD"] > 5].copy()
    moving["speed_band"] = pd.cut(
        moving["SPD"], bins=[5, 40, 60, 80, 200], labels=["5-40", "40-60", "60-80", "80+"]
    )
    moving["acc_bin"] = pd.qcut(moving["ACC"], 5, duplicates="drop")
    cells = (
        moving.groupby(["speed_band", "acc_bin", "ECO_bin"], observed=True)[config.TARGET]
        .agg(["mean", "count"])
        .reset_index()
    )
    means = cells.pivot(index=["speed_band", "acc_bin"], columns="ECO_bin", values="mean")
    counts = cells.pivot(index=["speed_band", "acc_bin"], columns="ECO_bin", values="count")
    means = means.reindex(columns=[0, 1])
    counts = counts.reindex(columns=[0, 1])
    means.columns = ["eco_off_mean", "eco_on_mean"]
    counts.columns = ["eco_off_n", "eco_on_n"]
    matched = pd.concat([means, counts], axis=1).dropna()
    if matched.empty:
        raise ValueError("No speed/pedal cell contains both ECO states")

    matched["delta"] = matched["eco_on_mean"] - matched["eco_off_mean"]
    matched["weight"] = np.minimum(matched["eco_off_n"], matched["eco_on_n"])
    matched = matched.reset_index()

    summary = []
    for band, part in matched.groupby("speed_band", observed=True):
        summary.append(
            {
                "band": str(band),
                "delta": float(np.average(part["delta"], weights=part["weight"])),
                "n": int(part["weight"].sum()),
            }
        )
    overall = float(np.average(matched["delta"], weights=matched["weight"]))
    return pd.DataFrame(summary), overall


def plot_eco_regime(
    frame: pd.DataFrame,
    trip: int = 4,
    directory: Path | None = None,
) -> Path:
    """ECO mode: a within-driver comparison at matched speed and pedal travel.

    A raw comparison of current in ECO-on vs ECO-off samples is confounded: the
    driver accelerates harder (mean pedal 27 % vs 12 %) and the ECO windows sit
    on slightly uphill sections. The right panel therefore compares samples only
    inside the same speed band *and* the same accelerator-pedal quintile, where
    the only systematic difference left is the mode itself. The numbers behind
    it come from :func:`matched_eco_summary`.
    """
    subset = frame.loc[frame[config.GROUP_COL] == trip].sort_values(config.TIME_COL)
    summary, overall = matched_eco_summary(frame, trip)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.55, 1]})

    axis = axes[0]
    elapsed = (subset[config.TIME_COL] - subset[config.TIME_COL].min()).dt.total_seconds() / 60
    axis.plot(elapsed, subset[config.TARGET], color=COLORS["current"], linewidth=0.7)
    axis.set_ylim(bottom=min(-60, subset[config.TARGET].min() - 5))

    eco_on = subset["ECO_bin"].fillna(0).to_numpy() == 1
    bottom, top = axis.get_ylim()
    axis.fill_between(
        elapsed,
        bottom,
        top,
        where=eco_on,
        color=COLORS["eco"],
        alpha=0.15,
        step="mid",
        label="ECO active",
    )
    axis.axhline(0, color="black", linewidth=0.6)
    axis.set_xlabel("Elapsed time [min]")
    axis.set_ylabel("Battery current [A]")
    axis.set_title(f"Trip {trip}: current trace while switching ECO mode")
    axis.legend(loc="upper right", fontsize=8)

    axis = axes[1]
    colours = [COLORS["eco"] if value < 0 else COLORS["current"] for value in summary["delta"]]
    bars = axis.bar(summary["band"], summary["delta"], color=colours, width=0.6)
    axis.axhline(0, color="black", linewidth=0.8)
    for bar, value, count in zip(bars, summary["delta"], summary["n"]):
        offset = -2.0 if value < 0 else 1.5
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.0f} A\nn={count}",
            ha="center",
            va="top" if value < 0 else "bottom",
            fontsize=7,
        )
    axis.set_ylim(min(summary["delta"]) * 1.35, 12)
    axis.set_xlabel("Speed band [km/h]")
    axis.set_ylabel("Current change [A]\n(ECO on - ECO off)")
    axis.set_title("Matched speed band and pedal quintile")

    fig.suptitle(
        "ECO mode lowers current once speed and accelerator pedal are held fixed "
        f"(matched mean {overall:+.0f} A)\n"
        "ECO-on windows also contain harder acceleration (mean pedal 27 % vs 12 %), "
        "so the raw trace above is confounded",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return save_figure(fig, "fig04_eco_regime.png", directory)
