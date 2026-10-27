"""Data cleaning and validation for the EV telemetry dataset.

Every decision taken here is recorded in a :class:`CleaningReport` so that the
data-quality section of the README can be generated from the code instead of
being typed by hand.

Cleaning principles used in this project
----------------------------------------
1. Never invent a measurement. Rows that cannot be trusted as measurements are
   removed and counted; they are not filled with guessed values.
2. Flag first, drop only when necessary. A corrupt GPS fix does not prevent the
   speed or current from being used, so it is flagged (``gps_valid == False``)
   rather than deleted.
3. Keep real gaps. Missing time intervals are preserved; interpolation across a
   transmission dropout would fabricate driving that never happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import config

# Raw hardware codes -> binary flags (documented in the dataset description).
ECO_MAP = {0: 0, 192: 1}
AIR_MAP = {0: 0, 2: 1}


@dataclass
class CleaningReport:
    """Counts produced by :func:`clean_telemetry`.

    The object is intentionally a plain data container: it can be printed,
    turned into a table, or asserted on inside tests.
    """

    n_raw: int = 0
    n_invalid_timestamp: int = 0
    n_out_of_window: int = 0
    n_duplicate_timestamps: int = 0
    n_inconsistent_timestamps: int = 0
    n_hard_invalid: int = 0
    n_clean: int = 0
    missing_before: pd.Series = field(default_factory=pd.Series)
    sentinel_counts: dict[str, int] = field(default_factory=dict)
    invalid_counts: dict[str, int] = field(default_factory=dict)
    unknown_flag_codes: dict[str, int] = field(default_factory=dict)
    gap_summary: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_removed(self) -> int:
        return self.n_raw - self.n_clean

    @property
    def removed_pct(self) -> float:
        return 100.0 * self.n_removed / self.n_raw if self.n_raw else 0.0


def build_timestamp(frame: pd.DataFrame) -> pd.Series:
    """Merge the six UTC columns (Y, M, D, H, MIN, SEC) into one UTC timestamp.

    The ``Y`` column stores a two-digit year (``23`` means 2023), so years below
    100 are shifted by 2000. Impossible component values (month 49, hour 83,
    ...) yield ``NaT`` instead of raising, and are reported by the caller.

    Returns
    -------
    pandas.Series
        Timezone-aware UTC timestamps; ``NaT`` where the components are invalid.
    """
    year = pd.to_numeric(frame["Y"], errors="coerce")
    year = year.where(year >= 100, year + config.YEAR_FLOOR)

    components = pd.DataFrame(
        {
            "year": year,
            "month": pd.to_numeric(frame["M"], errors="coerce"),
            "day": pd.to_numeric(frame["D"], errors="coerce"),
            "hour": pd.to_numeric(frame["H"], errors="coerce"),
            "minute": pd.to_numeric(frame["MIN"], errors="coerce"),
            "second": pd.to_numeric(frame["SEC"], errors="coerce"),
        },
        index=frame.index,
    )

    # pandas rolls an out-of-range hour over into the next day (hour 83 becomes
    # three days later) instead of failing, so the components are validated
    # explicitly before they are assembled.
    valid = (
        components["year"].between(1990, 2100)
        & components["month"].between(1, 12)
        & components["day"].between(1, 31)
        & components["hour"].between(0, 23)
        & components["minute"].between(0, 59)
        & components["second"].between(0, 59)
    )

    placeholder = pd.DataFrame(
        {
            "year": 2000,
            "month": 1,
            "day": 1,
            "hour": 0,
            "minute": 0,
            "second": 0,
        },
        index=frame.index,
    ).astype("float64")
    safe = components.where(valid, placeholder)

    timestamps = pd.to_datetime(safe, errors="coerce", utc=True)
    return timestamps.where(valid)


def convert_imu_to_si(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert raw MEMS counts to SI units and keep the raw values untouched.

    Accelerometer: ``raw / 2048 LSB/g * 9.80665 m/s^2``
    Gyroscope:     ``raw / 16.4 LSB/(deg/s)``

    Returns a copy with ``*_mps2`` and ``*_dps`` columns added.
    """
    out = frame.copy()
    for axis in ("AX", "AY", "AZ"):
        out[f"{axis}_mps2"] = out[axis] / config.ACC_LSB_PER_G * config.G
    for axis in ("GX", "GY", "GZ"):
        out[f"{axis}_dps"] = out[axis] / config.GYRO_LSB_PER_DPS
    return out


def standardize_flags(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Map hardware codes to binary 0/1 flags (ECO, AIR).

    Unknown codes are mapped to ``NaN`` and counted, so an unexpected hardware
    value is visible instead of being silently coerced to 0.
    """
    out = frame.copy()
    out["ECO_bin"] = out["ECO"].map(ECO_MAP)
    out["AIR_bin"] = out["AIR"].map(AIR_MAP)

    unknown = {
        "ECO": int((out["ECO_bin"].isna()).sum()),
        "AIR": int((out["AIR_bin"].isna()).sum()),
    }
    return out, unknown


def add_validity_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag physically implausible channels without deleting the rows.

    ``ALT <= 0`` is a sentinel for a missing GPS altitude (the route is driven
    at 850-1350 m), and ``VOL`` only returns a plausible pack voltage on two of
    the nine trips, so it is flagged rather than trusted blindly.
    """
    out = frame.copy()
    vol_lo, vol_hi = config.VALID_RANGES["VOL"]
    lat_lo, lat_hi = config.VALID_RANGES["LAT"]
    lon_lo, lon_hi = config.VALID_RANGES["LON"]

    out["vol_valid"] = out["VOL"].between(vol_lo, vol_hi)
    out["gps_valid"] = out["LAT"].between(lat_lo, lat_hi) & out["LON"].between(lon_lo, lon_hi)
    out["alt_missing"] = out["ALT"] <= 0
    return out


def _check_ranges(frame: pd.DataFrame) -> dict[str, int]:
    """Count values outside the plausible range for each hard-checked column."""
    counts: dict[str, int] = {}
    for column in config.HARD_RANGE_CHECKS:
        low, high = config.VALID_RANGES[column]
        counts[column] = int((~frame[column].between(low, high)).sum())
    return counts


def _gap_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe sampling gaps per trip (never filled, only reported)."""
    rows = []
    for trip, group in frame.groupby(config.GROUP_COL, sort=True):
        times = group[config.TIME_COL].sort_values()
        deltas = times.diff().dt.total_seconds().dropna()
        gaps = deltas[deltas > config.MAX_GAP_SECONDS]
        rows.append(
            {
                "COND": trip,
                "n_rows": int(len(group)),
                "start_utc": times.iloc[0],
                "end_utc": times.iloc[-1],
                "median_dt_s": float(deltas.median()) if len(deltas) else float("nan"),
                "n_gaps": int(len(gaps)),
                "largest_gap_s": float(gaps.max()) if len(gaps) else 0.0,
                "total_gap_s": float(gaps.sum()) if len(gaps) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _flag_inconsistent_timestamps(frame: pd.DataFrame, tolerance_s: float = 120.0) -> pd.Series:
    """Mark rows whose timestamp disagrees with the recording order.

    ``id`` increases with the odometer in every trip (verified: 100 % monotonic),
    so it is a reliable ordering key. A handful of rows carry a corrupted
    hour/minute (e.g. 01:00:00 in the middle of an afternoon trip); they are
    detected by comparing each timestamp with the median of its eleven nearest
    neighbours *in recording order*. Genuine gaps are not flagged, because every
    sample after a gap agrees with its own neighbours.
    """
    ordered = frame.sort_values([config.GROUP_COL, "id"], kind="mergesort")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    # total_seconds() is independent of the stored datetime resolution
    # (pandas may keep microseconds instead of nanoseconds).
    seconds = (ordered[config.TIME_COL] - epoch).dt.total_seconds()
    local_median = seconds.groupby(ordered[config.GROUP_COL]).transform(
        lambda series: series.rolling(11, center=True, min_periods=1).median()
    )
    return (seconds - local_median).abs() > tolerance_s


def clean_telemetry(frame: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the full cleaning pipeline and return the cleaned table plus a report.

    Steps, in order
    ---------------
    1. Merge the UTC date/time columns into ``timestamp``.
    2. Drop rows whose timestamp cannot be parsed, or falls outside the
       documented collection window (October 2023).
    3. Drop rows whose timestamp contradicts the recording order (corrupted
       hour/minute fields).
    4. Sort chronologically per trip and drop duplicated timestamps.
    5. Drop rows that violate hard physical ranges (speed, pedals, SOC, current).
    6. Convert IMU counts to SI units and standardise ECO/AIR flags.
    7. Flag unreliable channels (GPS, pack voltage, altitude) without dropping.
    """
    report = CleaningReport(n_raw=int(len(frame)))
    report.missing_before = frame.isna().sum()

    work = frame.copy()
    work[config.TIME_COL] = build_timestamp(work)

    invalid_ts = work[config.TIME_COL].isna()
    report.n_invalid_timestamp = int(invalid_ts.sum())
    work = work.loc[~invalid_ts]

    start_year, end_year = config.VALID_YEAR_RANGE
    in_window = (work[config.TIME_COL] >= pd.Timestamp(f"{start_year}-01-01", tz="UTC")) & (
        work[config.TIME_COL] < pd.Timestamp(f"{end_year}-01-01", tz="UTC")
    )
    report.n_out_of_window = int((~in_window).sum())
    work = work.loc[in_window]

    inconsistent = _flag_inconsistent_timestamps(work)
    report.n_inconsistent_timestamps = int(inconsistent.sum())
    work = work.loc[~inconsistent]

    work = work.sort_values([config.GROUP_COL, config.TIME_COL], kind="mergesort")

    before_dedup = len(work)
    work = work.drop_duplicates(subset=[config.GROUP_COL, config.TIME_COL], keep="first")
    report.n_duplicate_timestamps = int(before_dedup - len(work))

    report.invalid_counts = _check_ranges(work)
    hard_mask = pd.Series(True, index=work.index)
    for column in config.HARD_RANGE_CHECKS:
        low, high = config.VALID_RANGES[column]
        hard_mask &= work[column].between(low, high)
    report.n_hard_invalid = int((~hard_mask).sum())
    work = work.loc[hard_mask]

    work = convert_imu_to_si(work)
    work, report.unknown_flag_codes = standardize_flags(work)
    work = add_validity_flags(work)

    report.sentinel_counts = {
        "ALT <= 0 (missing altitude)": int(work["alt_missing"].sum()),
        "VOL outside 250-420 V": int((~work["vol_valid"]).sum()),
        "GPS outside Brazil bbox": int((~work["gps_valid"]).sum()),
    }

    work = work.reset_index(drop=True)
    report.n_clean = int(len(work))
    report.gap_summary = _gap_summary(work)
    return work, report


def add_elapsed_time(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``dt_s`` (seconds since the previous sample) and ``elapsed_s`` per trip.

    The first sample of each trip has ``dt_s = NaN`` by definition.
    """
    out = frame.copy()
    out["dt_s"] = out.groupby(config.GROUP_COL)[config.TIME_COL].diff().dt.total_seconds()
    out["elapsed_s"] = out.groupby(config.GROUP_COL)["dt_s"].cumsum()
    return out
