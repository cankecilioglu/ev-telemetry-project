"""Turn a :class:`~src.cleaning.CleaningReport` into the data-quality report.

The report is written to ``results/tables/`` both as CSV (for the report) and as
Markdown (for copy-paste into the README / slides).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .cleaning import CleaningReport
from .reporting import markdown_table


def build_quality_table(report: CleaningReport) -> pd.DataFrame:
    """Create the one-row-per-issue data-quality table."""
    removed = report.n_removed
    rows = [
        ("Raw rows read", report.n_raw, "Starting point of the pipeline."),
        (
            "Unparseable UTC timestamp",
            report.n_invalid_timestamp,
            "Y/M/D/H/MIN/SEC contained impossible components (e.g. month 49). Row dropped: the time of the observation cannot be established.",
        ),
        (
            "Timestamp outside 2023",
            report.n_out_of_window,
            "Two-digit year 0 (-> 2000) or corrupted date. Dropped for the same reason.",
        ),
        (
            "Duplicate (trip, timestamp)",
            report.n_duplicate_timestamps,
            "Same trip recorded twice at the same second; first occurrence kept.",
        ),
        (
            "Timestamp inconsistent with recording order",
            report.n_inconsistent_timestamps,
            "id order is 100 % monotonic with the odometer, so it defines the true sequence. Rows stamped 16-20 h away from their neighbours (corrupted hour/minute) were dropped.",
        ),
        (
            "Physically impossible values",
            report.n_hard_invalid,
            "Speed outside 0-200 km/h, pedals outside 0-100 %, SOC outside 0-100 %, current outside -200..400 A. Dropped.",
        ),
        (
            "Missing cells in raw file",
            int(report.missing_before.sum()),
            "The file has no NaN cells; missing data is encoded with sentinel values instead (see next rows).",
        ),
        (
            "ALT == 0 (missing altitude)",
            report.sentinel_counts.get("ALT <= 0 (missing altitude)", 0),
            "Route is driven between 850 and 1350 m, so 0 means 'no GPS fix'. Flagged (alt_missing), not deleted.",
        ),
        (
            "VOL outside 250-420 V",
            report.sentinel_counts.get("VOL outside 250-420 V", 0),
            "The pack-voltage channel is corrupted on 7 of 9 trips; rows kept but flagged (vol_valid).",
        ),
        (
            "GPS outside Brazil bounding box",
            report.sentinel_counts.get("GPS outside Brazil bbox", 0),
            "Sign bit lost / zeroed fixes. Flagged (gps_valid), not deleted; GPS is not a model input.",
        ),
    ]
    table = pd.DataFrame(rows, columns=["Issue", "Count", "Decision"])
    table.loc[len(table)] = [
        "Rows remaining after cleaning",
        report.n_clean,
        f"{report.removed_pct:.2f} % of raw rows removed.",
    ]
    return table


def build_trip_table(clean: pd.DataFrame) -> pd.DataFrame:
    """Per-trip row counts, time span and observed sampling interval."""
    rows = []
    for trip, group in clean.groupby(config.GROUP_COL, sort=True):
        deltas = group[config.TIME_COL].diff().dt.total_seconds().dropna()
        rows.append(
            {
                "COND": int(trip),
                "rows": int(len(group)),
                "start_utc": group[config.TIME_COL].min(),
                "end_utc": group[config.TIME_COL].max(),
                "duration_min": round((group[config.TIME_COL].max() - group[config.TIME_COL].min()).total_seconds() / 60, 1),
                "median_dt_s": round(float(deltas.median()), 2) if len(deltas) else float("nan"),
                "distance_km": round(float(group[config.ODOMETER].max() - group[config.ODOMETER].min()), 1),
            }
        )
    return pd.DataFrame(rows)


def write_quality_report(
    report: CleaningReport,
    quality_table: pd.DataFrame,
    trip_table: pd.DataFrame,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Write the data-quality tables as CSV and Markdown. Returns both paths."""
    output_dir = Path(output_dir or config.TABLE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "data_quality_report.csv"
    md_path = output_dir / "data_quality_report.md"

    quality_table.to_csv(csv_path, index=False)

    lines = ["# Data-quality report", "", markdown_table(quality_table), ""]
    lines += ["## Sampling gaps per trip", "", "Gaps are reported, never interpolated.", ""]
    if not report.gap_summary.empty:
        lines += [markdown_table(report.gap_summary, floatfmt=".1f"), ""]
    lines += ["## Trip summary after cleaning", "", markdown_table(trip_table, floatfmt=".1f"), ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return csv_path, md_path
