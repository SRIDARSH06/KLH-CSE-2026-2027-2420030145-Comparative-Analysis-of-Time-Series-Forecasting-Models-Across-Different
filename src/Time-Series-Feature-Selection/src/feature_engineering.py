"""Leakage-safe, small feature set for one-step-ahead forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_GROUPS = {
    "Lag": ["lag_1", "lag_7", "lag_14", "lag_28"],
    "Rolling": ["rolling_mean_7", "rolling_mean_14", "rolling_std_7", "rolling_std_14"],
    "Calendar": ["day_of_week", "month", "week_of_year", "is_weekend"],
    "Change": ["change_1", "change_7", "growth_7"],
}

FEATURE_DESCRIPTIONS = {
    "lag_1": "The previous observation.",
    "lag_7": "The value seven observations ago.",
    "lag_14": "The value fourteen observations ago.",
    "lag_28": "The value twenty-eight observations ago.",
    "rolling_mean_7": "Mean of the seven observations immediately before the target.",
    "rolling_mean_14": "Mean of the fourteen observations immediately before the target.",
    "rolling_std_7": "Variation across the seven observations immediately before the target.",
    "rolling_std_14": "Variation across the fourteen observations immediately before the target.",
    "day_of_week": "Day number of the target date, from Monday=0 to Sunday=6.",
    "month": "Calendar month of the target date.",
    "week_of_year": "ISO calendar week of the target date.",
    "is_weekend": "1 when the target date is Saturday or Sunday, otherwise 0.",
    "change_1": "Change between the two observations immediately before the target.",
    "change_7": "Change between the last observation and the observation seven periods before it.",
    "growth_7": "Relative seven-period change, with safe handling for zero denominators.",
}

ALL_FEATURES = [feature for features in FEATURE_GROUPS.values() for feature in features]
BASIC_FEATURES = ["lag_1", "lag_7", "rolling_mean_7", "day_of_week", "month", "is_weekend"]


def create_features(series: pd.DataFrame) -> pd.DataFrame:
    """Build features for target y_t using observations strictly before t."""
    if not {"date", "value"}.issubset(series.columns):
        raise ValueError("Feature engineering expects date and value columns.")
    values = series["value"].astype(float)
    dates = pd.to_datetime(series["date"])
    features = pd.DataFrame({"date": dates, "target": values})
    previous = values.shift(1)

    for lag in (1, 7, 14, 28):
        features[f"lag_{lag}"] = values.shift(lag)
    for window in (7, 14):
        features[f"rolling_mean_{window}"] = previous.rolling(window, min_periods=window).mean()
        features[f"rolling_std_{window}"] = previous.rolling(window, min_periods=window).std()

    features["day_of_week"] = dates.dt.dayofweek
    features["month"] = dates.dt.month
    features["week_of_year"] = dates.dt.isocalendar().week.astype(float)
    features["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(int)

    features["change_1"] = values.shift(1) - values.shift(2)
    features["change_7"] = values.shift(1) - values.shift(8)
    denominator = values.shift(8).replace(0, np.nan)
    features["growth_7"] = (values.shift(1) - values.shift(8)) / denominator
    return features