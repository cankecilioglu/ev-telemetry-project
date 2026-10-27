"""Reading the raw telemetry file with explicit, human-readable failures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


class DataLoadError(RuntimeError):
    """Raised when the telemetry file cannot be read or is structurally wrong."""


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the raw EV telemetry CSV.

    Parameters
    ----------
    path:
        Path to the CSV file. Relative paths are resolved against the current
        working directory, which keeps the project portable.

    Returns
    -------
    pandas.DataFrame
        The unmodified table read from disk.

    Raises
    ------
    DataLoadError
        If the file is missing, unreadable, empty, or does not contain the
        columns the rest of the pipeline depends on.
    """
    csv_path = Path(path)

    if not csv_path.exists():
        raise DataLoadError(
            f"Data file not found: {csv_path.resolve()}\n"
            "Download 'dataset.csv' from the Kaggle dataset "
            "'marceloreis76/electric-vehicle-telemetry-dataset' and place it "
            "under data/raw/, then run:\n"
            "    python main.py --data data/raw/dataset.csv"
        )

    if not csv_path.is_file():
        raise DataLoadError(f"Expected a file but got a directory: {csv_path.resolve()}")

    try:
        frame = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(f"Data file is empty: {csv_path.resolve()}") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(
            f"Could not parse the CSV file: {csv_path.resolve()}\n"
            f"Pandas reported: {exc}"
        ) from exc
    except OSError as exc:
        raise DataLoadError(
            f"Could not read {csv_path.resolve()}: {exc}. "
            "Close the file if it is open in another program and try again."
        ) from exc

    if frame.empty:
        raise DataLoadError(f"Data file contains no rows: {csv_path.resolve()}")

    missing = [column for column in config.REQUIRED_RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise DataLoadError(
            "The data file is missing required columns: "
            + ", ".join(missing)
            + "\nExpected columns: "
            + ", ".join(config.REQUIRED_RAW_COLUMNS)
        )

    return frame
