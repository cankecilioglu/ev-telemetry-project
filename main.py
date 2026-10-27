"""Command-line entry point.

Examples
--------
    python main.py --data data/raw/dataset.csv
    python main.py --data data/raw/dataset.csv --stage clean
    python main.py --data data/raw/dataset.csv --stage model --no-tune

Stages
------
clean     clean the raw file, write the data-quality report
explore   clean + exploratory figures
baseline  clean + chronological split + training-mean baseline
model     clean + features + physics calibration + Random Forest + error analysis
all       everything above (default)
"""

from __future__ import annotations

import argparse
import random
import sys

import numpy as np
import pandas as pd

from src import config, diagnostics, exploration, modeling, quality
from src.cleaning import add_elapsed_time, clean_telemetry
from src.features import build_features
from src.io import DataLoadError, load_raw_data
from src.plotting import apply_style
from src.reporting import save_table

METHODS = ("baseline", "linear", "physics", "rf")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EV telemetry analysis: cleaning, exploration and modelling."
    )
    parser.add_argument(
        "--data",
        default=str(config.RAW_DATA_DIR / "dataset.csv"),
        help="Path to the raw telemetry CSV (default: data/raw/dataset.csv).",
    )
    parser.add_argument(
        "--stage",
        choices=["clean", "explore", "baseline", "model", "all"],
        default="all",
        help="Pipeline stage to run (default: all).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=config.TEST_FRACTION,
        help="Fraction of the latest observations used for testing (default: 0.2).",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip the Random Forest grid search (fast smoke run).",
    )
    return parser.parse_args(argv)


def set_seeds(seed: int = config.RANDOM_SEED) -> None:
    """Fix every random seed used by the pipeline for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def report_cleaning(report) -> None:
    """Print the human-readable cleaning summary."""
    print("\n== Cleaning summary ==")
    print(f"  raw rows          : {report.n_raw}")
    print(f"  invalid timestamps: {report.n_invalid_timestamp}")
    print(f"  outside 2023      : {report.n_out_of_window}")
    print(f"  inconsistent time : {report.n_inconsistent_timestamps}")
    print(f"  duplicate stamps  : {report.n_duplicate_timestamps}")
    print(f"  impossible values : {report.n_hard_invalid}  {report.invalid_counts}")
    print(f"  rows remaining    : {report.n_clean}  ({report.removed_pct:.2f} % removed)")
    for label, count in report.sentinel_counts.items():
        print(f"  flagged           : {label}: {count}")
    if not report.gap_summary.empty:
        gaps = report.gap_summary
        print(
            "  sampling gaps     : "
            f"{int(gaps['n_gaps'].sum())} gaps > {config.MAX_GAP_SECONDS:.0f} s "
            f"(largest {gaps['largest_gap_s'].max():.0f} s)"
        )


def run_clean(data_path: str):
    """Load and clean the raw file; write the quality tables."""
    raw = load_raw_data(data_path)
    print(f"\nLoaded {len(raw):,} rows x {raw.shape[1]} columns from {data_path}")

    clean, report = clean_telemetry(raw)
    clean = add_elapsed_time(clean)
    report_cleaning(report)

    quality_table = quality.build_quality_table(report)
    trip_table = quality.build_trip_table(clean)
    csv_path, md_path = quality.write_quality_report(report, quality_table, trip_table)

    print("\n== Data-quality table ==")
    print(quality_table.to_string(index=False))
    print(f"\n  saved: {csv_path}")
    print(f"  saved: {md_path}")
    return clean, report


def run_explore(clean) -> None:
    """Create the exploratory figures."""
    print("\n== Figures ==")
    for path in (
        exploration.plot_trip_timeseries(clean, trip=1),
        exploration.plot_voltage_channel(clean),
        exploration.plot_pedal_response(clean),
        exploration.plot_eco_regime(clean, trip=4),
    ):
        print(f"  saved: {path}")


def run_baseline(clean, test_fraction: float) -> None:
    """Chronological split and the training-mean baseline."""
    print("\n== Chronological split and baseline ==")
    split = modeling.chronological_split(clean, test_fraction=test_fraction)
    composition = modeling.split_composition(split)

    print(f"  training rows : {len(split.train):,}")
    print(f"  test rows     : {len(split.test):,}")
    print(f"  cut timestamp : {split.split_timestamp}")
    print(composition.to_string(index=False))

    train_mean = float(split.train[config.TARGET].mean())
    predictions = modeling.baseline_predict(split.train, split.test)
    metrics = modeling.regression_metrics(split.test[config.TARGET], predictions)
    metrics_table = pd.DataFrame(
        [
            {
                "model": "baseline_train_mean",
                "features": "(none)",
                **{key: value for key, value in metrics.items() if key != "n"},
                "n_test": metrics["n"],
            }
        ]
    )

    print(f"\n  training-mean prediction: {train_mean:.2f} A")
    print(
        f"  test MAE = {metrics['MAE']:.2f} A | RMSE = {metrics['RMSE']:.2f} A "
        f"| bias = {metrics['bias']:.2f} A | R2 = {metrics['R2']:.3f}"
    )

    save_table(composition, "split_composition", "Chronological split composition")
    save_table(metrics_table, "baseline_metrics", "Baseline (training-mean) metrics")


def run_model(clean, test_fraction: float, tune_rf: bool = True) -> None:
    """Physics calibration, Random Forest, comparison and error analysis."""
    print("\n== Modelling ==")
    features = build_features(clean)

    split = modeling.chronological_split(features, test_fraction=test_fraction)
    print(f"  training rows : {len(split.train):,}")
    print(f"  test rows     : {len(split.test):,}")
    print(f"  cut timestamp : {split.split_timestamp}")

    metrics, predictions, extras = modeling.evaluate_models(split, tune_rf=tune_rf)
    calibration = extras["calibration"]

    print("\n  Test performance (same held-out rows):")
    print(metrics[["method", "MAE_A", "RMSE_A", "bias_A", "R2", "n_test"]].to_string(index=False))

    print(f"\n  Physics calibration fitted on {calibration.n_fit_rows:,} training rows:")
    print(calibration.interpretation().to_string(index=False))

    if extras["rf_cv_mae"] is not None:
        print(f"\n  Random Forest: CV MAE = {extras['rf_cv_mae']:.2f} A")
    print(f"  best parameters: {extras['rf_params']}")
    print(f"  feature importance: {extras['rf_importances']}")

    # ---- robustness check: every driver in the test set -------------------
    print("\n== Robustness check: per-trip chronological split ==")
    per_trip = modeling.per_trip_split(features, test_fraction=test_fraction)
    metrics_pt, predictions_pt, _ = modeling.evaluate_models(per_trip, tune_rf=False)
    print(metrics_pt[["method", "MAE_A", "bias_A", "n_test"]].to_string(index=False))

    # ---- error analysis ---------------------------------------------------
    analysed = modeling.add_operating_regime(predictions)
    error_speed = modeling.error_table(analysed, "speed_band", METHODS)
    error_regime = modeling.error_table(analysed, "regime", METHODS)
    error_grade = modeling.error_table(analysed, "grade_band", METHODS)

    print("\n  MAE by operating regime [A]:")
    print(error_regime.pivot(index="regime", columns="method", values="MAE_A").round(2).to_string())

    # ---- outputs ----------------------------------------------------------
    save_table(metrics, "model_comparison", "Model comparison (global chronological split)")
    save_table(metrics_pt, "model_comparison_per_trip", "Model comparison (per-trip split)")
    save_table(calibration.interpretation(), "physics_coefficients", "Physics calibration coefficients")
    save_table(error_speed, "error_by_speed", "MAE by speed band")
    save_table(error_regime, "error_by_regime", "MAE by operating regime")
    save_table(error_grade, "error_by_grade", "MAE by grade band")
    # Row-level predictions are a working file, not a report: CSV only.
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = config.TABLE_DIR / "test_predictions.csv"
    analysed.to_csv(predictions_path, index=False)
    print(f"  saved: {predictions_path}")

    print("\n== Model figures ==")
    test_trip = int(predictions[config.GROUP_COL].iloc[0])
    for path in (
        diagnostics.plot_model_comparison(metrics),
        diagnostics.plot_error_analysis(error_speed, error_regime),
        diagnostics.plot_prediction_window(predictions, trip=test_trip, minutes=25),
    ):
        print(f"  saved: {path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_seeds()
    apply_style()

    try:
        clean, _ = run_clean(args.data)
        if args.stage in ("explore", "all"):
            run_explore(clean)
        if args.stage in ("baseline", "all"):
            run_baseline(clean, args.test_fraction)
        if args.stage in ("model", "all"):
            run_model(clean, args.test_fraction, tune_rf=not args.no_tune)
    except DataLoadError as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 2

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
