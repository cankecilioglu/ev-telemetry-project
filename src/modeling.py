"""Time-ordered validation, the three estimation methods and error analysis.

Models compared (all evaluated on the same test rows with MAE in ampere):

1. ``baseline``  - predict the training-set mean.
2. ``linear``    - optional simple reference: linear regression on speed and
                   acceleration only (allowed by the instructor's feedback).
3. ``physics``   - longitudinal power balance calibrated by least squares on
                   the training set; coefficients are physical parameters.
4. ``rf``        - RandomForestRegressor, tuned with time-series cross validation
                   inside the training set only.

No hyper-parameter, scaler or imputer ever sees the test set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from . import config
from .features import MODEL_FEATURES, PHYSICS_FEATURES, build_physics_design


@dataclass
class ChronologicalSplit:
    """Container for the time-ordered train/test slices."""

    train: pd.DataFrame
    test: pd.DataFrame
    split_timestamp: pd.Timestamp

    @property
    def fraction(self) -> float:
        return len(self.train) / (len(self.train) + len(self.test))


def chronological_split(
    frame: pd.DataFrame,
    test_fraction: float = config.TEST_FRACTION,
    time_col: str = config.TIME_COL,
) -> ChronologicalSplit:
    """Split the table by time: first (1 - test_fraction) rows train, rest test.

    The frame is sorted by timestamp; the cut is made on row order, so the test
    set is strictly *later* than the training set. Nothing is shuffled.
    """
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    if len(frame) < 10:
        raise ValueError("Not enough rows to split (need at least 10).")

    ordered = frame.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    n_test = int(round(len(ordered) * test_fraction))
    n_test = min(max(n_test, 1), len(ordered) - 1)
    cut = len(ordered) - n_test

    train = ordered.iloc[:cut].copy()
    test = ordered.iloc[cut:].copy()
    return ChronologicalSplit(train=train, test=test, split_timestamp=ordered[time_col].iloc[cut])


def per_trip_split(
    frame: pd.DataFrame,
    test_fraction: float = config.TEST_FRACTION,
) -> ChronologicalSplit:
    """Robustness check: the last ``test_fraction`` of *each* trip is held out.

    This guarantees that every driver appears once in the test set, which the
    global chronological split cannot offer (it puts only the last day in test).
    Used as a secondary evaluation, not as the primary one.
    """
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    train_parts, test_parts = [], []
    for _, group in frame.groupby(config.GROUP_COL, sort=True):
        group = group.sort_values(config.TIME_COL)
        n_test = max(1, int(round(len(group) * test_fraction)))
        train_parts.append(group.iloc[:-n_test])
        test_parts.append(group.iloc[-n_test:])

    train = pd.concat(train_parts).sort_values(config.TIME_COL).reset_index(drop=True)
    test = pd.concat(test_parts).sort_values(config.TIME_COL).reset_index(drop=True)
    return ChronologicalSplit(train=train, test=test, split_timestamp=test[config.TIME_COL].min())


def split_composition(split: ChronologicalSplit) -> pd.DataFrame:
    """Rows per trip in each slice - shows which trips end up in the test set."""
    rows = []
    for trip in sorted(set(split.train[config.GROUP_COL]).union(split.test[config.GROUP_COL])):
        rows.append(
            {
                "COND": int(trip),
                "train_rows": int((split.train[config.GROUP_COL] == trip).sum()),
                "test_rows": int((split.test[config.GROUP_COL] == trip).sum()),
            }
        )
    table = pd.DataFrame(rows)
    total = pd.DataFrame(
        [{"COND": "total", "train_rows": len(split.train), "test_rows": len(split.test)}]
    )
    return pd.concat([table, total], ignore_index=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE (primary), RMSE, signed bias and R^2 in the target unit."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "bias": float(residual.mean()),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "n": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Method 1: mean baseline
# ---------------------------------------------------------------------------
def baseline_predict(train: pd.DataFrame, test: pd.DataFrame, target: str = config.TARGET) -> np.ndarray:
    """Predict the training-set mean for every test observation."""
    return np.full(len(test), float(train[target].mean()))


# ---------------------------------------------------------------------------
# Method 2: physics-based calibration
# ---------------------------------------------------------------------------
@dataclass
class PhysicsCalibration:
    """Fitted coefficients of the longitudinal power balance."""

    beta: np.ndarray
    feature_names: list[str]
    n_fit_rows: int

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        design = build_physics_design(frame).fillna(0.0)
        return design[self.feature_names].to_numpy() @ self.beta

    def interpretation(self) -> pd.DataFrame:
        """Translate the electrical coefficients back into vehicle parameters."""
        eta = config.VEHICLE["drivetrain_efficiency"]
        rho = config.VEHICLE["air_density_kg_m3"]
        g = config.G
        beta = dict(zip(self.feature_names, self.beta))

        apparent_mass = beta["traction"] * eta
        cd_a = beta["aero"] * eta * 2.0 / rho
        gravity_mass = beta["grade"] * eta / g
        rolling_coefficient = (
            beta["rolling"] * eta / (apparent_mass * g) if apparent_mass > 0 else np.nan
        )

        rows = [
            ("m / eta  [kg]", beta["traction"], config.VEHICLE["mass_kg"] / eta, "apparent mass"),
            ("rho*Cd*A / (2 eta)", beta["aero"], 1.2 * 0.32 * 2.16 / (2 * eta), "aerodynamic drag"),
            ("m*g / eta  [N]", beta["grade"], config.VEHICLE["mass_kg"] * g / eta, "grade force"),
            ("Crr*m*g / eta  [N]", beta["rolling"], 0.012 * config.VEHICLE["mass_kg"] * g / eta, "rolling resistance"),
            ("P_base  [W]", beta["aux_base"], np.nan, "base electrical load"),
            ("P_air  [W]", beta["aux_air"], np.nan, "air-conditioning load"),
        ]
        table = pd.DataFrame(rows, columns=["Coefficient", "Fitted", "Prior", "Meaning"])
        table["Implied"] = [
            f"{apparent_mass:.0f} kg",
            f"CdA = {cd_a:.3f} m^2",
            f"{gravity_mass:.0f} kg",
            f"Crr = {rolling_coefficient:.4f}",
            "",
            "",
        ]
        return table


def fit_physics_calibration(train: pd.DataFrame, target: str = config.TARGET) -> PhysicsCalibration:
    """Least-squares calibration of the power balance on the training rows."""
    design = build_physics_design(train)[PHYSICS_FEATURES]
    target_values = train[target]
    valid = design.notna().all(axis=1) & target_values.notna()

    beta, *_ = np.linalg.lstsq(
        design.loc[valid].to_numpy(), target_values.loc[valid].to_numpy(), rcond=None
    )
    return PhysicsCalibration(beta=beta, feature_names=list(PHYSICS_FEATURES), n_fit_rows=int(valid.sum()))


# ---------------------------------------------------------------------------
# Method 3: machine learning
# ---------------------------------------------------------------------------
def build_rf_pipeline(seed: int = config.RANDOM_SEED) -> Pipeline:
    """Median imputation + Random Forest (the imputer is fitted on train only)."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestRegressor(random_state=seed, n_jobs=-1)),
        ]
    )


def fit_random_forest(
    train: pd.DataFrame,
    features: list[str] | None = None,
    tune: bool = True,
    target: str = config.TARGET,
) -> tuple[Pipeline, dict, float | None]:
    """Fit the Random Forest, optionally tuning it with time-series CV.

    Returns the fitted pipeline, the parameter dictionary and the best
    cross-validated MAE (``None`` when tuning is disabled).
    """
    features = features or MODEL_FEATURES
    pipeline = build_rf_pipeline()

    if not tune:
        pipeline.set_params(rf__n_estimators=300, rf__max_depth=12, rf__min_samples_leaf=1)
        pipeline.fit(train[features], train[target])
        return pipeline, pipeline.get_params(), None

    search = GridSearchCV(
        pipeline,
        param_grid={f"rf__{key}": values for key, values in config.RF_PARAM_GRID.items()},
        cv=TimeSeriesSplit(n_splits=config.CV_SPLITS),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    search.fit(train[features], train[target])
    best = {key.replace("rf__", ""): value for key, value in search.best_params_.items()}
    return search.best_estimator_, best, float(-search.best_score_)


def fit_linear_reference(
    train: pd.DataFrame,
    features: tuple[str, ...] = ("SPD", "accel_mps2"),
    target: str = config.TARGET,
) -> Pipeline:
    """Optional reference model: linear regression on speed and acceleration only."""
    pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("linear", LinearRegression())]
    )
    pipeline.fit(train[list(features)], train[target])
    return pipeline


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_models(
    split: ChronologicalSplit,
    tune_rf: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit every method on the training slice and score them on the test slice.

    Returns
    -------
    metrics : DataFrame   one row per method, MAE/RMSE/bias/R^2
    predictions : DataFrame   aligned per-row predictions for error analysis
    extras : dict   fitted objects and diagnostics (coefficients, RF params, ...)
    """
    train, test = split.train, split.test
    y_true = test[config.TARGET].to_numpy(dtype=float)

    calibration = fit_physics_calibration(train)
    rf_model, rf_params, rf_cv_mae = fit_random_forest(train, tune=tune_rf)
    linear_model = fit_linear_reference(train)

    predictions = pd.DataFrame(
        {
            "COND": test[config.GROUP_COL].to_numpy(),
            config.TIME_COL: test[config.TIME_COL].to_numpy(),
            "y_true": y_true,
            "baseline": baseline_predict(train, test),
            "linear": linear_model.predict(test[["SPD", "accel_mps2"]]),
            "physics": calibration.predict(test),
            "rf": rf_model.predict(test[MODEL_FEATURES]),
            # context columns, kept for the error analysis
            "SPD": test["SPD"].to_numpy(),
            "ACC": test["ACC"].to_numpy(),
            "BRK": test["BRK"].to_numpy(),
            "ECO_bin": test["ECO_bin"].to_numpy(),
            "AIR_bin": test["AIR_bin"].to_numpy(),
            "grade_pct": test["grade_pct"].to_numpy(),
        }
    )

    labels = {
        "baseline": ("training mean", "-"),
        "linear": ("linear regression", "SPD, acceleration"),
        "physics": ("physics calibration", "a*v, v^3, grade, v, AIR"),
        "rf": ("random forest", ", ".join(MODEL_FEATURES)),
    }

    rows = []
    for method, (name, features) in labels.items():
        metrics = regression_metrics(y_true, predictions[method])
        rows.append(
            {
                "method": name,
                "key": method,
                "features": features,
                "MAE_A": round(metrics["MAE"], 3),
                "RMSE_A": round(metrics["RMSE"], 3),
                "bias_A": round(metrics["bias"], 3),
                "R2": round(metrics["R2"], 4),
                "n_test": metrics["n"],
            }
        )
    metrics_table = pd.DataFrame(rows)

    extras = {
        "calibration": calibration,
        "rf_params": rf_params,
        "rf_cv_mae": rf_cv_mae,
        "rf_importances": dict(
            zip(MODEL_FEATURES, rf_model.named_steps["rf"].feature_importances_.round(4))
        ),
        "train_mean": float(train[config.TARGET].mean()),
    }
    return metrics_table, predictions, extras


SPEED_BANDS = ["0-20", "20-40", "40-60", "60-80", "80-100", "100+"]
GRADE_BANDS = ["downhill", "flat", "uphill"]
REGIMES = ["accelerating", "braking", "coasting"]
CATEGORY_ORDER = {"speed_band": SPEED_BANDS, "grade_band": GRADE_BANDS, "regime": REGIMES}


def add_operating_regime(frame: pd.DataFrame) -> pd.DataFrame:
    """Label each row with speed band, grade band and pedal regime."""
    out = frame.copy()
    out["speed_band"] = pd.cut(
        out["SPD"],
        bins=[-0.1, 20, 40, 60, 80, 100, 200],
        labels=SPEED_BANDS,
    )
    out["grade_band"] = pd.cut(
        out["grade_pct"],
        bins=[-15.1, -2, 2, 15.1],
        labels=GRADE_BANDS,
    )
    out["regime"] = np.select(
        [out["ACC"] > 0, out["BRK"] > 0],
        ["accelerating", "braking"],
        default="coasting",
    )
    return out


def error_table(predictions: pd.DataFrame, by: str, methods: tuple[str, ...]) -> pd.DataFrame:
    """MAE and bias per category, for every method, on the same rows."""
    rows = []
    for category, group in predictions.groupby(by, observed=True):
        for method in methods:
            residual = group[method] - group["y_true"]
            rows.append(
                {
                    by: str(category),
                    "method": method,
                    "n": int(len(group)),
                    "MAE_A": round(float(residual.abs().mean()), 2),
                    "bias_A": round(float(residual.mean()), 2),
                }
            )
    table = pd.DataFrame(rows)

    # Keep the physical order of the bins (alphabetical order would put "100+"
    # right after "0-20" in plots and reports).
    if by in CATEGORY_ORDER:
        present = [label for label in CATEGORY_ORDER[by] if label in set(table[by])]
        table[by] = pd.Categorical(table[by], categories=present, ordered=True)
        table = table.sort_values([by, "method"], kind="mergesort").reset_index(drop=True)
    return table
