"""Exploratory analysis tables used by the Streamlit interface."""

from __future__ import annotations

import pandas as pd


def descriptive_statistics(series: pd.DataFrame) -> pd.DataFrame:
    return series["value"].describe().to_frame("value").round(4)


def rolling_values(series: pd.DataFrame) -> pd.DataFrame:
    output = series.copy()
    output["rolling_mean_7"] = output["value"].rolling(7, min_periods=1).mean()
    output["rolling_mean_14"] = output["value"].rolling(14, min_periods=1).mean()
    return output


def temporal_pattern(series: pd.DataFrame) -> pd.DataFrame:
    result = series.copy()
    result["day_of_week"] = result["date"].dt.day_name()
    result["day_number"] = result["date"].dt.dayofweek
    grouped = result.groupby(["day_number", "day_of_week"], as_index=False)["value"].mean()
    return grouped.sort_values("day_number").rename(columns={"value": "mean_value"})