"""Unit tests for the cleaning pipeline.

Run with:
    pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.cleaning import (
    add_validity_flags,
    build_timestamp,
    clean_telemetry,
    convert_imu_to_si,
    standardize_flags,
)


def make_raw_frame(rows: int = 5) -> pd.DataFrame:
    """Small synthetic table with the same schema as the Kaggle file."""
    return pd.DataFrame(
        {
            "id": np.arange(1_000_000_000, 1_000_000_000 + rows),
            "COND": [1] * rows,
            "LAT": [-22.9] * rows,
            "LON": [-43.2] * rows,
            "ALT": [1000.0] * rows,
            "AX": [0] * rows,
            "AY": [0] * rows,
            "AZ": [2048] * rows,
            "GX": [0] * rows,
            "GY": [0] * rows,
            "GZ": [0] * rows,
            "Y": [23] * rows,
            "M": [10] * rows,
            "D": [23] * rows,
            "H": [12] * rows,
            "MIN": [0] * rows,
            "SEC": list(range(rows)),
            "CH": [80.0] * rows,
            "VOL": [355] * rows,
            "CUR": [20] * rows,
            "SPD": [40] * rows,
            "ODO": [26000.0 + i for i in range(rows)],
            "BRK": [0] * rows,
            "ACC": [15] * rows,
            "AUT": [180] * rows,
            "ECO": [0] * rows,
            "AIR": [2] * rows,
        }
    )


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------
def test_build_timestamp_uses_two_digit_year():
    frame = make_raw_frame(3)
    timestamps = build_timestamp(frame)

    assert str(timestamps.dt.tz) == "UTC"
    assert timestamps.iloc[0] == pd.Timestamp("2023-10-23 12:00:00", tz="UTC")
    assert timestamps.iloc[2] == pd.Timestamp("2023-10-23 12:00:02", tz="UTC")


def test_build_timestamp_returns_nat_for_impossible_components():
    frame = make_raw_frame(2)
    frame.loc[0, "M"] = 49   # month 49 exists in the real file
    frame.loc[1, "H"] = 83   # so does hour 83

    timestamps = build_timestamp(frame)

    assert timestamps.isna().sum() == 2


# ---------------------------------------------------------------------------
# Unit conversion and flags
# ---------------------------------------------------------------------------
def test_imu_conversion_to_si_units():
    frame = make_raw_frame(1)
    frame["GX"] = frame["GX"].astype(float)
    frame.loc[0, ["AX", "AZ"]] = [2048, -2048]
    frame.loc[0, "GX"] = 16.4

    converted = convert_imu_to_si(frame)

    assert converted.loc[0, "AX_mps2"] == pytest.approx(config.G, rel=1e-9)
    assert converted.loc[0, "AZ_mps2"] == pytest.approx(-config.G, rel=1e-9)
    assert converted.loc[0, "GX_dps"] == pytest.approx(1.0, rel=1e-9)


def test_hardware_codes_become_binary_flags():
    frame = make_raw_frame(3)
    frame["ECO"] = [0, 192, 0]
    frame["AIR"] = [0, 2, 0]

    flagged, unknown = standardize_flags(frame)

    assert flagged["ECO_bin"].tolist() == [0, 1, 0]
    assert flagged["AIR_bin"].tolist() == [0, 1, 0]
    assert unknown == {"ECO": 0, "AIR": 0}


def test_unknown_hardware_code_is_not_silently_zeroed():
    frame = make_raw_frame(1)
    frame["ECO"] = [99]

    flagged, unknown = standardize_flags(frame)

    assert pd.isna(flagged.loc[0, "ECO_bin"])
    assert unknown["ECO"] == 1


# ---------------------------------------------------------------------------
# Cleaning decisions
# ---------------------------------------------------------------------------
def test_clean_telemetry_drops_bad_rows_and_reports_them():
    frame = make_raw_frame(6)

    frame.loc[1, "M"] = 49          # impossible timestamp  -> dropped
    frame.loc[2, "Y"] = 0           # year 2000             -> out of window
    frame.loc[3, "SEC"] = 0         # duplicate timestamp   -> dropped
    frame.loc[4, "SPD"] = 500       # impossible speed      -> dropped

    clean, report = clean_telemetry(frame)

    assert report.n_raw == 6
    assert report.n_invalid_timestamp == 1
    assert report.n_out_of_window == 1
    assert report.n_duplicate_timestamps == 1
    assert report.n_hard_invalid == 1
    assert report.n_clean == 2
    assert len(clean) == 2


def test_clean_telemetry_drops_timestamp_contradicting_recording_order():
    frame = make_raw_frame(21)
    frame.loc[10, "H"] = 1          # 01:00:00 in the middle of an afternoon trip
    frame.loc[10, "MIN"] = 0
    frame.loc[10, "SEC"] = 0

    clean, report = clean_telemetry(frame)

    assert report.n_inconsistent_timestamps == 1
    assert len(clean) == 20
    assert (clean["timestamp"].dt.hour != 1).all()


def test_clean_telemetry_flags_sentinels_without_deleting_rows():
    frame = make_raw_frame(2)
    frame.loc[0, "ALT"] = 0.0      # missing altitude sentinel
    frame.loc[0, "VOL"] = 22       # corrupted voltage channel
    frame.loc[1, "LAT"] = 22.0     # GPS sign bit lost

    clean, _ = clean_telemetry(frame)

    assert len(clean) == 2
    assert bool(clean.loc[0, "alt_missing"]) is True
    assert bool(clean.loc[0, "vol_valid"]) is False
    assert bool(clean.loc[1, "gps_valid"]) is False


def test_clean_telemetry_is_sorted_per_trip():
    frame = make_raw_frame(10)
    shuffled = frame.sample(frac=1.0, random_state=0)

    clean, _ = clean_telemetry(shuffled)

    times = clean.loc[clean["COND"] == 1, "timestamp"].tolist()
    assert times == sorted(times)


def test_voltage_flag_marks_only_plausible_pack_voltages():
    frame = make_raw_frame(3)
    frame["VOL"] = [355, 22, 0]

    flagged = add_validity_flags(frame)

    assert flagged["vol_valid"].tolist() == [True, False, False]


def test_gap_summary_preserves_missing_seconds():
    frame = make_raw_frame(6)
    frame.loc[3:, "SEC"] = [30, 31, 32]   # 25 s hole between sample 2 and 3

    _, report = clean_telemetry(frame)

    trip = report.gap_summary.iloc[0]
    assert trip["n_gaps"] == 1
    assert trip["largest_gap_s"] == pytest.approx(28.0)
