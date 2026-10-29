"""Dataset discovery, loading, validation, and benchmark format handling."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO
import json
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONFIG_PATH = ROOT / "config.json"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _matches_pattern(path: Path, pattern: str) -> bool:
    return path.match(pattern) or path.name.lower().startswith(pattern.replace("*", "").lower())


def discover_dataset_files(config: dict | None = None) -> dict[str, Path]:
    """Find files in data/ using the patterns in config.json."""
    config = config or load_config()
    DATA_DIR.mkdir(exist_ok=True)
    discovered: dict[str, Path] = {}
    files = [path for path in DATA_DIR.iterdir() if path.is_file()]
    for dataset in config.get("datasets", []):
        for pattern in dataset.get("file_patterns", []):
            matches = [path for path in files if _matches_pattern(path, pattern)]
            if matches:
                discovered[dataset["id"]] = sorted(matches)[0]
                break
    return discovered


def _read_path(path: Path, sheet_name: str | None = None, raw: bool = False) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, header=None if raw else 0)
    if suffix in {".xlsx", ".xlsm"}:
        try:
            return pd.read_excel(path, sheet_name=sheet_name or 0, header=None if raw else 0)
        except ValueError:
            # NN5 files use several worksheet names in the wild; the first sheet
            # is a useful safe fallback when the configured name is absent.
            return pd.read_excel(path, sheet_name=0, header=None if raw else 0)
    raise ValueError("Only CSV and XLSX/XLSM files are supported.")


def _read_uploaded(file_bytes: bytes, filename: str, sheet_name: str | None = None, raw: bool = False) -> pd.DataFrame:
    buffer: BinaryIO = BytesIO(file_bytes)
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(buffer, header=None if raw else 0)
    if suffix in {".xlsx", ".xlsm"}:
        try:
            return pd.read_excel(buffer, sheet_name=sheet_name or 0, header=None if raw else 0)
        except ValueError:
            buffer.seek(0)
            return pd.read_excel(buffer, sheet_name=0, header=None if raw else 0)
    raise ValueError("Only CSV and XLSX/XLSM files are supported.")


def read_uploaded_table(
    file_bytes: bytes,
    filename: str,
    sheet_name: str | None = None,
    raw: bool = False,
) -> pd.DataFrame:
    """Public table reader used by dataset-specific adapters."""
    return _read_uploaded(file_bytes, filename, sheet_name=sheet_name, raw=raw)


def _date_score(values: pd.Series) -> float:
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    return float(parsed.notna().mean())


def detect_date_column(frame: pd.DataFrame) -> str:
    """Detect a date/time column, preferring columns whose names say date or time."""
    best_column: str | None = None
    best_score = 0.0
    for column in frame.columns:
        name = str(column).lower()
        score = _date_score(frame[column])
        if re.search(r"date|time|timestamp|datetime", name):
            score += 0.15
        if score > best_score:
            best_column, best_score = str(column), score
    if best_column is None or best_score < 0.55:
        raise ValueError("Could not detect a date/time column. Check that the file contains dates.")
    return best_column


def _friendly_series_name(value: object, fallback: str) -> str:
    if pd.isna(value) or str(value).strip() in {"", "nan", "None"}:
        return fallback
    text = str(value).strip()
    return text if len(text) <= 80 else fallback


def _normalize_wide_frame(raw: pd.DataFrame, dataset_id: str, raw_nn5: bool = False) -> pd.DataFrame:
    """Return a consistent wide frame with a date column followed by numeric series."""
    if not raw_nn5:
        frame = raw.copy()
        frame.columns = [str(column).strip() for column in frame.columns]
        date_column = detect_date_column(frame)
        date_values = pd.to_datetime(frame[date_column], errors="coerce")
        output = pd.DataFrame({"date": date_values})
        for column in frame.columns:
            if column == date_column:
                continue
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.notna().sum() >= 3:
                output[str(column)] = numeric
        return output.dropna(subset=["date"]).reset_index(drop=True)

    # NN5 workbooks often have notes above the observation table and no header row.
    date_column_index: int | None = None
    best_score = 0.0
    for index in raw.columns:
        score = _date_score(raw[index])
        if score > best_score:
            date_column_index, best_score = index, score
    if date_column_index is None or best_score < 0.45:
        raise ValueError("The NN5 workbook did not contain a recognizable date column.")

    dates = pd.to_datetime(raw[date_column_index], errors="coerce", format="mixed")
    valid_rows = dates.notna()
    output = pd.DataFrame({"date": dates.loc[valid_rows].reset_index(drop=True)})
    first_data_row = int(np.flatnonzero(valid_rows.to_numpy())[0])
    series_number = 1
    for column in raw.columns:
        if column == date_column_index:
            continue
        numeric = pd.to_numeric(raw[column], errors="coerce")
        values = numeric.loc[valid_rows].reset_index(drop=True)
        if values.notna().sum() < 3:
            continue
        label = None
        for row_index in range(max(0, first_data_row - 8), first_data_row):
            candidate = raw.iloc[row_index][column]
            if isinstance(candidate, str) and candidate.strip() and "nn5" in candidate.lower():
                label = candidate.strip()
        label = _friendly_series_name(label, f"NN5-{series_number:03d}")
        output[label] = values
        series_number += 1
    return output.reset_index(drop=True)


def load_dataset(path: Path, dataset_id: str = "auto", sheet_name: str | None = None) -> pd.DataFrame:
    is_nn5 = dataset_id.lower() == "nn5"
    raw = _read_path(path, sheet_name=sheet_name if is_nn5 else None, raw=is_nn5)
    return _normalize_wide_frame(raw, dataset_id, raw_nn5=is_nn5)


def load_uploaded_dataset(file_bytes: bytes, filename: str, dataset_id: str = "auto", sheet_name: str | None = None) -> pd.DataFrame:
    is_nn5 = dataset_id.lower() == "nn5"
    raw = _read_uploaded(file_bytes, filename, sheet_name=sheet_name if is_nn5 else None, raw=is_nn5)
    return _normalize_wide_frame(raw, dataset_id, raw_nn5=is_nn5)


def validate_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate and summarize a normalized wide frame."""
    if "date" not in frame.columns:
        raise ValueError("The normalized dataset is missing its date column.")
    series_columns = [column for column in frame.columns if column != "date"]
    if not series_columns:
        raise ValueError("No numeric time-series columns were found.")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    for column in series_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    summary = {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "series_count": len(series_columns),
        "series_columns": series_columns,
        "start": frame["date"].min(),
        "end": frame["date"].max(),
        "missing_by_series": frame[series_columns].isna().sum().to_dict(),
        "missing_total": int(frame[series_columns].isna().sum().sum()),
    }
    return frame, summary


def prepare_series(frame: pd.DataFrame, series_name: str) -> tuple[pd.DataFrame, dict]:
    """Create a clean, single-series frame while preserving missing-value diagnostics."""
    if series_name not in frame.columns:
        raise ValueError(f"Series '{series_name}' was not found.")
    series = frame[["date", series_name]].rename(columns={series_name: "value"}).copy()
    series["value"] = pd.to_numeric(series["value"], errors="coerce")
    missing_before = int(series["value"].isna().sum())
    series = series.set_index("date")
    series["value"] = series["value"].interpolate(method="time").ffill().bfill()
    series = series.reset_index()
    if series["value"].notna().sum() < 40:
        raise ValueError("At least 40 usable observations are recommended for the forecasting experiment.")
    return series, {
        "missing_before": missing_before,
        "missing_after": int(series["value"].isna().sum()),
        "observations": int(series.shape[0]),
    }