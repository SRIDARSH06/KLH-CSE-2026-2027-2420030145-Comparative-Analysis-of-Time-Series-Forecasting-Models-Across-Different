"""Small, transparent preprocessing helpers for time-series data."""

from __future__ import annotations

import pandas as pd

from .data_loader import prepare_series, validate_frame


def validate_and_prepare(frame: pd.DataFrame, series_name: str) -> tuple[pd.DataFrame, dict]:
    validated, summary = validate_frame(frame)
    series, series_summary = prepare_series(validated, series_name)
    summary.update(series_summary)
    return series, summary