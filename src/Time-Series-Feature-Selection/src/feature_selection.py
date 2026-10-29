"""Training-only redundant-feature removal and Random Forest ranking."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .feature_engineering import ALL_FEATURES


@dataclass
class SelectionResult:
    candidate_features: list[str]
    retained_after_correlation: list[str]
    redundant_features: list[str]
    importance: pd.DataFrame
    selected_features: list[str]


def remove_correlated_features(
    train_frame: pd.DataFrame,
    features: list[str],
    threshold: float = 0.92,
) -> tuple[list[str], list[str]]:
    correlations = train_frame[features].corr().abs()
    retained: list[str] = []
    redundant: list[str] = []
    for feature in features:
        if any(correlations.loc[feature, previous] > threshold for previous in retained):
            redundant.append(feature)
        else:
            retained.append(feature)
    return retained, redundant


def select_features(
    feature_frame: pd.DataFrame,
    train_end: int,
    correlation_threshold: float = 0.92,
    top_n: int = 8,
    n_estimators: int = 80,
) -> SelectionResult:
    usable = feature_frame.dropna(subset=ALL_FEATURES + ["target"]).reset_index(drop=True)
    train = usable.iloc[:train_end].copy()
    if len(train) < 20:
        raise ValueError("There are not enough training observations for feature selection.")

    retained, redundant = remove_correlated_features(train, ALL_FEATURES, correlation_threshold)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(train[retained], train["target"])
    importance = pd.DataFrame({"feature": retained, "importance": model.feature_importances_})
    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
    selected = importance.head(min(top_n, len(importance)))["feature"].tolist()
    return SelectionResult(ALL_FEATURES, retained, redundant, importance, selected)