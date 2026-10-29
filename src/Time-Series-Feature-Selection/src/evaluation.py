"""Forecast metrics with safe handling of zeros."""

from __future__ import annotations

import numpy as np


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.abs(actual) + np.abs(predicted)
    values = np.where(denominator == 0, 0, 2 * np.abs(predicted - actual) / denominator)
    return float(np.mean(values) * 100)


def metric_dict(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {"MAE": mae(actual, predicted), "RMSE": rmse(actual, predicted), "sMAPE": smape(actual, predicted)}