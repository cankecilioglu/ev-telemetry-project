"""Unit tests for gap markers and the matched ECO comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.exploration import insert_gap_markers, matched_eco_summary


def test_gap_marker_breaks_the_line_without_resampling():
    frame = pd.DataFrame(
        {
            config.GROUP_COL: 1,
            config.TIME_COL: pd.to_datetime(
                ["2023-10-23 08:00:00", "2023-10-23 08:00:02", "2023-10-23 08:01:02"],
                utc=True,
            ),
            "SPD": [10.0, 12.0, 20.0],
        }
    )

    marked = insert_gap_markers(frame, max_gap_s=10.0)

    assert len(marked) == 4  # three measurements + one NaN marker
    assert marked["SPD"].isna().sum() == 1
    marker_time = marked.loc[marked["SPD"].isna(), config.TIME_COL].iloc[0]
    assert marker_time == pd.Timestamp("2023-10-23 08:00:32", tz="UTC")
    assert marked.loc[marked["SPD"].notna(), "SPD"].tolist() == [10.0, 12.0, 20.0]


def test_matched_eco_summary_removes_the_pedal_confound():
    rng = np.random.default_rng(4)
    pedal = rng.uniform(0, 60, 400)  # identical pedal profile in both modes
    rows = []
    for eco in (0, 1):
        # Same demand, but ECO on saves a fixed 18 A; because the pedal profile
        # is identical, every matched cell must show exactly -18 A.
        current = 0.8 * pedal + (0 if eco == 0 else -18.0)
        rows.append(
            pd.DataFrame(
                {
                    config.GROUP_COL: 4,
                    config.TIME_COL: pd.date_range(
                        "2023-10-24 14:20:00", periods=400, freq="2s", tz="UTC"
                    ),
                    "SPD": 50.0,
                    "ACC": pedal,
                    "CUR": current,
                    "ECO_bin": float(eco),
                }
            )
        )
    frame = pd.concat(rows, ignore_index=True)

    summary, overall = matched_eco_summary(frame, trip=4)

    assert overall == pytest.approx(-18.0, abs=0.01)
    assert (summary["delta"] < 0).all()
    assert summary["n"].sum() > 0
