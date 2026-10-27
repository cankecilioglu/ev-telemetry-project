"""Derived engineering variables and the model design matrices.

Design decisions
----------------
* Longitudinal acceleration is differentiated from the speed channel rather than
  read from ``AX``. The MEMS accelerometer in this dataset has a per-trip offset
  between 0.07 and 3.13 m/s^2 and correlates with the differentiated speed at
  only r = 0.14, while the differentiated speed correlates with the battery
  current at r = 0.50 (``AX`` only reaches r = 0.22). This is a data-quality
  finding, not a modelling convenience - see the README.
* Road grade is estimated over a trailing ~30 s window (15 samples), so the
  feature only uses information that is already available at the sampling
  instant.
* Every new column states its unit in its name.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Simultaneous-measurement features used by the machine-learning model.
MODEL_FEATURES = [
    "SPD",           # km/h
    "accel_mps2",    # m/s^2, from dSPD/dt
    "ACC",           # % accelerator pedal
    "BRK",           # % brake pedal
    "ECO_bin",       # 0/1 economy mode
    "AIR_bin",       # 0/1 air conditioning
    "grade_pct",     # % road grade
]

# Basis functions of the longitudinal power balance. Each column already
# contains the 1/V factor, so the fitted coefficient is a physical parameter.
PHYSICS_FEATURES = ["traction", "aero", "grade", "rolling", "aux_base", "aux_air"]

GRADE_WINDOW = 15          # samples, ~30 s at 1-2 s sampling
MIN_GRADE_DISTANCE_KM = 0.05
MAX_GRADE = 0.15           # +/- 15 % is already steeper than any public road


def add_kinematics(frame: pd.DataFrame) -> pd.DataFrame:
    """Add speed in m/s and battery power in W.

    ``power_W`` is only defined where the pack-voltage channel is trustworthy
    (``vol_valid``); ``power_nominal_W`` uses the nominal pack voltage and is
    therefore defined for every row.
    """
    out = frame.copy()
    out["speed_mps"] = out["SPD"] / 3.6
    out["power_W"] = np.where(out["vol_valid"], out["VOL"] * out["CUR"], np.nan)
    out["power_nominal_W"] = config.VEHICLE["nominal_pack_voltage_v"] * out["CUR"]
    return out


def add_longitudinal_acceleration(frame: pd.DataFrame) -> pd.DataFrame:
    """Differentiate the speed channel per trip: ``a = dSPD/dt`` in m/s^2.

    The first sample of every trip has no predecessor and stays ``NaN``.
    """
    out = frame.copy()
    timestamps = out[config.TIME_COL]
    dt = timestamps.groupby(out[config.GROUP_COL]).diff().dt.total_seconds()
    dv = out["SPD"].groupby(out[config.GROUP_COL]).diff() / 3.6

    with np.errstate(divide="ignore", invalid="ignore"):
        acceleration = np.where(dt > 0, dv.to_numpy() / dt.to_numpy(), np.nan)

    out["accel_mps2"] = acceleration
    return out


def add_road_grade(frame: pd.DataFrame, window: int = GRADE_WINDOW) -> pd.DataFrame:
    """Estimate road grade (%) from GPS altitude over a trailing distance window.

    Altitude has 1 m resolution, so a single-sample difference is pure
    quantisation noise; the change over ``window`` samples (~30 s, a few hundred
    metres) is what carries the slope signal. The sign convention is
    positive = uphill.
    """
    out = frame.copy()
    groups = out[config.GROUP_COL]
    altitude = out["ALT"].where(out["ALT"] > 0)          # 0 is a sentinel

    delta_alt = altitude - altitude.groupby(groups).shift(window)
    delta_distance = out[config.ODOMETER] - out[config.ODOMETER].groupby(groups).shift(window)

    valid = delta_distance > MIN_GRADE_DISTANCE_KM
    with np.errstate(divide="ignore", invalid="ignore"):
        grade = np.where(valid, delta_alt / (delta_distance * 1000.0), np.nan)

    out["grade_pct"] = np.clip(grade, -MAX_GRADE, MAX_GRADE) * 100.0
    return out


def build_physics_design(frame: pd.DataFrame, voltage: float | None = None) -> pd.DataFrame:
    """Build the six basis functions of the longitudinal power balance.

    ``I_batt = [m*a*v + 1/2 rho Cd A v^3 + m g sin(theta) v + Crr m g v] / (eta V)
    + P_aux/V``

    is linear in the coefficients ``m/eta``, ``rho Cd A/(2 eta)``, ``m g/eta``,
    ``Crr m g/eta``, ``P_base`` and ``P_air``. Dividing by the pack voltage makes
    every coefficient an *apparent electrical* parameter, which is what the
    measurement actually sees.
    """
    pack_voltage = voltage or config.VEHICLE["nominal_pack_voltage_v"]
    speed = frame["SPD"] / 3.6
    acceleration = frame["accel_mps2"]
    theta = np.arctan(frame["grade_pct"] / 100.0)

    return pd.DataFrame(
        {
            "traction": acceleration * speed / pack_voltage,
            "aero": speed**3 / pack_voltage,
            "grade": np.sin(theta) * speed / pack_voltage,
            "rolling": speed / pack_voltage,
            "aux_base": 1.0 / pack_voltage,
            "aux_air": frame["AIR_bin"] / pack_voltage,
        },
        index=frame.index,
    )


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply every derived-variable step in the documented order."""
    return add_road_grade(add_longitudinal_acceleration(add_kinematics(frame)))
