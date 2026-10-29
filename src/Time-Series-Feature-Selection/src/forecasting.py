"""Chronological forecasting experiment using one Random Forest model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .evaluation import metric_dict
from .feature_engineering import ALL_FEATURES, BASIC_FEATURES
from .feature_selection import SelectionResult, select_features


@dataclass
class ExperimentResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    selection: SelectionResult
    train_rows: int
    test_rows: int
    feature_counts: dict[str, int]


def run_experiment(
    feature_frame: pd.DataFrame,
    test_size: float = 0.2,
    correlation_threshold: float = 0.92,
    top_n: int = 8,
    n_estimators: int = 80,
) -> ExperimentResult:
    usable = feature_frame.dropna(subset=ALL_FEATURES + ["target"]).reset_index(drop=True)
    if len(usable) < 50:
        raise ValueError("At least 50 complete feature rows are needed for a meaningful experiment.")
    split = int(len(usable) * (1 - test_size))
    split = max(25, min(split, len(usable) - 5))
    selection = select_features(
        usable,
        train_end=split,
        correlation_threshold=correlation_threshold,
        top_n=top_n,
        n_estimators=n_estimators,
    )
    actual = usable.iloc[split:]["target"].to_numpy(dtype=float)
    dates = usable.iloc[split:]["date"].reset_index(drop=True)
    naive_prediction = usable.iloc[split:]["lag_1"].to_numpy(dtype=float)
    predictions = pd.DataFrame({"date": dates, "actual": actual, "Naive": naive_prediction})
    feature_sets = {
        "Basic": BASIC_FEATURES,
        "All Engineered": ALL_FEATURES,
        "Selected Engineered": selection.selected_features,
    }
    model_metrics = [{"Feature Set": "Naive", "Number of Features": 0, **metric_dict(actual, naive_prediction)}]
    for name, features in feature_sets.items():
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        )
        model.fit(usable.iloc[:split][features], usable.iloc[:split]["target"])
        predicted = model.predict(usable.iloc[split:][features])
        predictions[name] = predicted
        model_metrics.append(
            {"Feature Set": name, "Number of Features": len(features), **metric_dict(actual, predicted)}
        )
    metrics = pd.DataFrame(model_metrics)
    return ExperimentResult(
        metrics=metrics,
        predictions=predictions,
        selection=selection,
        train_rows=split,
        test_rows=len(usable) - split,
        feature_counts={
            "candidate": len(ALL_FEATURES),
            "retained": len(selection.retained_after_correlation),
            "selected": len(selection.selected_features),
        },
    )