"""Unit tests for derived variables and the physics design matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import (
    PHYSICS_FEATURES,
    add_longitudinal_acceleration,
    add_road_grade,
    build_physics_design,
)


def make_frame(rows: int = 40) -> pd.DataFrame:
    stamps = pd.date_range("2023-10-23 08:00:00", periods=rows, freq="2s", tz="UTC")
    return pd.DataFrame(
        {
            "COND": [1] * rows,
            "timestamp": stamps,
            "SPD": np.linspace(0, 100, rows),
            "ODO": np.linspace(26000.0, 26001.0, rows),
            "ALT": np.linspace(1000.0, 1010.0, rows),
            "alt_missing": [False] * rows,
            "AIR_bin": [0] * rows,
            "grade_pct": [0.0] * rows,
            "accel_mps2": [0.0] * rows,
        }
    )


def test_acceleration_comes_from_the_speed_channel():
    frame = make_frame(6)
    frame["SPD"] = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]   # +10 km/h every 2 s

    result = add_longitudinal_acceleration(frame)

    expected = (10.0 / 3.6) / 2.0
    assert result["accel_mps2"].iloc[1] == pytest.approx(expected)
    assert np.isnan(result["accel_mps2"].iloc[0])


def test_grade_is_positive_uphill():
    frame = make_frame(40)
    frame["ALT"] = np.linspace(1000.0, 1100.0, 40)        # +100 m over 1 km
    frame["ODO"] = np.linspace(26000.0, 26001.0, 40)

    result = add_road_grade(frame, window=15)

    assert result["grade_pct"].iloc[-1] == pytest.approx(10.0, abs=0.5)


def test_grade_ignores_altitude_sentinel():
    frame = make_frame(40)
    frame.loc[5:10, "ALT"] = 0.0
    frame.loc[5:10, "alt_missing"] = True
    frame["ALT"] = frame["ALT"].where(frame["ALT"] > 0, np.nan)

    result = add_road_grade(frame, window=15)

    assert result["grade_pct"].isna().sum() > 0   # no fabricated slope


def test_physics_design_has_the_documented_basis_functions():
    frame = make_frame(10)
    frame["accel_mps2"] = 1.0
    frame["SPD"] = 36.0                                  # 10 m/s

    design = build_physics_design(frame)

    assert list(design.columns) == PHYSICS_FEATURES
    assert design["traction"].iloc[0] == pytest.approx(
        1.0 * 10.0 / config.VEHICLE["nominal_pack_voltage_v"]
    )
    assert design["aux_base"].iloc[0] == pytest.approx(1.0 / 350.0)
    assert np.isfinite(design.to_numpy()).all()
