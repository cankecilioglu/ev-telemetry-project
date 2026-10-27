"""Project-wide configuration.

All paths are derived from the repository root, so the project can be moved to
another machine (or cloned from GitHub) without editing any file.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the repository root, never absolute)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

for _directory in (DATA_DIR, RAW_DATA_DIR, RESULTS_DIR, FIGURE_DIR, TABLE_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
GROUP_COL = "COND"          # trip / driver identifier (1..9)
TIME_COL = "timestamp"      # merged UTC timestamp (created during cleaning)
TARGET = "CUR"              # battery current in ampere (signed: + discharge, - regen)
ODOMETER = "ODO"

# Columns that must exist in the raw CSV. A missing column is a hard error.
REQUIRED_RAW_COLUMNS = [
    "id", "COND", "LAT", "LON", "ALT",
    "AX", "AY", "AZ", "GX", "GY", "GZ",
    "Y", "M", "D", "H", "MIN", "SEC",
    "CH", "VOL", "CUR", "SPD", "ODO", "BRK", "ACC", "AUT", "ECO", "AIR",
]

# ---------------------------------------------------------------------------
# Sensor conversion constants (from the sensor data sheets)
# ---------------------------------------------------------------------------
G = 9.80665                 # standard gravity, m/s^2
ACC_LSB_PER_G = 2048.0      # accelerometer sensitivity, LSB/g
GYRO_LSB_PER_DPS = 16.4     # gyroscope sensitivity, LSB/(deg/s)

# ---------------------------------------------------------------------------
# Timestamp validation
# ---------------------------------------------------------------------------
# The Y column stores a two-digit year (23 -> 2023). Values outside this range
# are treated as corrupted, not as measurements.
YEAR_FLOOR = 2000
VALID_YEAR_RANGE = (2023, 2024)   # [start, end)

# ---------------------------------------------------------------------------
# Physically plausible ranges. Values outside them are flagged and, for the
# variables needed by the model, removed from the modelling table.
# ---------------------------------------------------------------------------
VALID_RANGES = {
    "SPD": (0.0, 200.0),        # km/h
    "ACC": (0.0, 100.0),        # % pedal travel
    "BRK": (0.0, 100.0),        # % pedal travel
    "CH": (0.0, 100.0),         # % state of charge
    "CUR": (-200.0, 400.0),     # A (signed)
    "VOL": (250.0, 420.0),      # V, only used as a validity flag
    "LAT": (-40.0, 10.0),       # deg (Brazil)
    "LON": (-80.0, -30.0),      # deg (Brazil)
}

# Rows failing these checks are dropped: they cannot be used as a target or as
# a predictor without inventing data.
HARD_RANGE_CHECKS = ["SPD", "ACC", "BRK", "CH", "CUR"]

# A gap longer than this many seconds is kept as a real gap in time-series
# plots and is never interpolated.
MAX_GAP_SECONDS = 10.0

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
TEST_FRACTION = 0.20
CV_SPLITS = 5
RF_PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [12, 20, None],
    "min_samples_leaf": [1, 3, 8],
}
RANDOM_SEED = 42

# Vehicle parameters used by the physics-based calibration.
# JAC iEV40 (a.k.a. iEV7S) public specifications; used as *priors* only.
VEHICLE = {
    "mass_kg": 1535.0,          # curb mass, kg
    "frontal_area_m2": 2.16,
    "drag_coefficient": 0.32,
    "air_density_kg_m3": 1.20,  # ~25 degC, sea level; Rio de Janeiro ~ 1000 m
    "rolling_resistance": 0.012,
    "drivetrain_efficiency": 0.90,
    "regen_efficiency": 0.65,
    "nominal_pack_voltage_v": 350.0,   # used where the VOL channel is corrupted
}

# Deterministic output settings
FIGURE_DPI = 150
