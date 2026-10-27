"""Small helpers for writing result tables without extra dependencies."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def markdown_table(frame: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table.

    Implemented locally so that the project does not depend on ``tabulate``.
    """
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else format(value, floatfmt)
            )

    header = "| " + " | ".join(str(column) for column in formatted.columns) + " |"
    separator = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
        for row in formatted.itertuples(index=False)
    ]
    return "\n".join([header, separator, *rows])


def save_table(frame: pd.DataFrame, name: str, title: str | None = None) -> tuple[Path, Path]:
    """Save ``frame`` as CSV and Markdown under ``results/tables``."""
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.TABLE_DIR / f"{name}.csv"
    md_path = config.TABLE_DIR / f"{name}.md"

    frame.to_csv(csv_path, index=False)
    body = markdown_table(frame)
    md_path.write_text((f"# {title}\n\n" if title else "") + body + "\n", encoding="utf-8")
    return csv_path, md_path
