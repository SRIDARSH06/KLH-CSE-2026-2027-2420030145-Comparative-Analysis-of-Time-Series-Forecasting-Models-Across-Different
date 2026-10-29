"""Matplotlib figures kept separate from the Streamlit page code."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def _base_figure(title: str, xlabel: str = "", ylabel: str = ""):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, ax


def plot_series(series: pd.DataFrame, title: str):
    fig, ax = _base_figure(title, "Date", "Value")
    ax.plot(series["date"], series["value"], color="#1f6f78", linewidth=1.4)
    fig.autofmt_xdate()
    return fig


def plot_distribution(series: pd.DataFrame, title: str):
    fig, ax = _base_figure(title, ylabel="Frequency")
    ax.hist(series["value"], bins=30, color="#e28f41", alpha=0.85, edgecolor="white")
    return fig


def plot_rolling(rolling: pd.DataFrame, title: str):
    fig, ax = _base_figure(title, "Date", "Value")
    ax.plot(rolling["date"], rolling["value"], color="#aab7b8", linewidth=0.8, label="Observed")
    ax.plot(rolling["date"], rolling["rolling_mean_7"], color="#1f6f78", label="Rolling mean (7)")
    ax.plot(rolling["date"], rolling["rolling_mean_14"], color="#d05a3a", label="Rolling mean (14)")
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    return fig


def plot_temporal(pattern: pd.DataFrame, title: str):
    fig, ax = _base_figure(title, "Day of week", "Mean value")
    ax.bar(pattern["day_of_week"], pattern["mean_value"], color="#1f6f78")
    ax.tick_params(axis="x", rotation=30)
    return fig


def plot_correlation(correlation: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(correlation, cmap="RdYlBu_r", vmin=-1, vmax=1)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=75, ha="right")
    ax.set_yticks(range(len(correlation.index)), correlation.index)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_importance(importance: pd.DataFrame, title: str):
    display = importance.sort_values("importance").tail(12)
    fig, ax = _base_figure(title, "Importance", "Feature")
    ax.barh(display["feature"], display["importance"], color="#d05a3a")
    return fig


def plot_forecasts(predictions: pd.DataFrame, title: str):
    fig, ax = _base_figure(title, "Date", "Value")
    ax.plot(predictions["date"], predictions["actual"], color="#1d2733", linewidth=2, label="Actual")
    for name, color in [("Naive", "#8e9aaf"), ("Basic", "#e28f41"), ("All Engineered", "#1f6f78"), ("Selected Engineered", "#d05a3a")]:
        ax.plot(predictions["date"], predictions[name], linewidth=1.1, label=name, alpha=0.9)
    ax.legend(frameon=False, ncol=2)
    fig.autofmt_xdate()
    return fig