"""Adapters that convert NN5, M4, and M5 inputs to one common series shape."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re

import pandas as pd

from .data_loader import read_uploaded_table


COMMON_COLUMNS = ["date", "series_id", "target"]
M5_METADATA_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def _read_csv_bytes(file_bytes: bytes, filename: str, **kwargs) -> pd.DataFrame:
    if Path(filename).suffix.lower() not in {".csv", ".txt"}:
        raise ValueError(f"{filename} must be a CSV file.")
    return pd.read_csv(BytesIO(file_bytes), **kwargs)


def _read_any_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    return read_uploaded_table(file_bytes, filename)


def _find_column(columns: list[object], candidates: set[str]) -> str | None:
    for column in columns:
        normalized = re.sub(r"[^a-z0-9]", "", str(column).lower())
        if normalized in candidates:
            return str(column)
    return None


def canonical_to_wide(canonical: pd.DataFrame) -> pd.DataFrame:
    """Convert common date/series_id/target rows to the app's wide frame."""
    required = set(COMMON_COLUMNS)
    if not required.issubset(canonical.columns):
        raise ValueError(f"Adapter output must contain {', '.join(COMMON_COLUMNS)}.")
    cleaned = canonical.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["target"] = pd.to_numeric(cleaned["target"], errors="coerce")
    cleaned["series_id"] = cleaned["series_id"].astype(str)
    cleaned = cleaned.dropna(subset=["date", "target"])
    if cleaned.empty:
        raise ValueError("The adapter found no usable observations.")
    return (
        cleaned.pivot_table(index="date", columns="series_id", values="target", aggfunc="first")
        .sort_index()
        .reset_index()
        .rename_axis(None, axis=1)
    )


def _sequential_periods(length: int) -> pd.DatetimeIndex:
    # Some official M4 training files provide ordered values but no calendar.
    # A labelled sequential index preserves order without changing targets.
    return pd.date_range("2000-01-01", periods=length, freq="D")


def _m4_long_columns(frame: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    columns = list(frame.columns)
    id_col = _find_column(columns, {"id", "uniqueid", "seriesid", "m4id"})
    date_col = _find_column(columns, {"date", "ds", "period", "timestamp", "datetime"})
    target_col = _find_column(columns, {"y", "target", "value", "sales", "demand"})
    return id_col, date_col, target_col


def _m4_row_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    id_col, _, _ = _m4_long_columns(frame)
    if id_col is None:
        first = frame.columns[0]
        first_numeric = pd.to_numeric(frame[first], errors="coerce").notna().mean()
        if first_numeric < 0.5:
            id_col = str(first)
    value_columns = [str(column) for column in frame.columns if str(column) != id_col]
    numeric_columns = [
        column for column in value_columns if pd.to_numeric(frame[column], errors="coerce").notna().sum() >= 2
    ]
    if not numeric_columns:
        raise ValueError(
            "Could not recognize M4 data. Expected long columns such as unique_id, date/period, and y, "
            "or one row per series with numeric period columns."
        )
    ids = (
        frame[id_col].astype(str).tolist()
        if id_col is not None
        else [f"M4-{index + 1:05d}" for index in range(len(frame))]
    )
    return frame[numeric_columns].apply(pd.to_numeric, errors="coerce"), ids


def m4_series_options(file_bytes: bytes, filename: str) -> list[str]:
    frame = _read_any_bytes(file_bytes, filename)
    id_col, date_col, target_col = _m4_long_columns(frame)
    if id_col and date_col and target_col:
        return frame[id_col].dropna().astype(str).drop_duplicates().tolist()
    _, ids = _m4_row_matrix(frame)
    if date_col:
        return [str(column) for column in frame.columns if str(column) != date_col and pd.to_numeric(frame[column], errors="coerce").notna().sum() >= 2]
    return ids


def load_m4_series(file_bytes: bytes, filename: str, series_id: str) -> tuple[pd.DataFrame, dict]:
    frame = _read_any_bytes(file_bytes, filename)
    id_col, date_col, target_col = _m4_long_columns(frame)
    if id_col and date_col and target_col:
        selected = frame[frame[id_col].astype(str) == str(series_id)].copy()
        if selected.empty:
            raise ValueError(f"M4 series '{series_id}' was not found.")
        canonical = pd.DataFrame(
            {
                "date": selected[date_col],
                "series_id": str(series_id),
                "target": selected[target_col],
            }
        )
        return canonical, {"format": "long", "period_index_is_sequential": False}

    if date_col:
        if series_id not in frame.columns:
            raise ValueError(f"M4 series '{series_id}' was not found among the numeric columns.")
        canonical = pd.DataFrame(
            {
                "date": frame[date_col],
                "series_id": str(series_id),
                "target": frame[series_id],
            }
        )
        return canonical, {"format": "dated-wide", "period_index_is_sequential": False}

    matrix, ids = _m4_row_matrix(frame)
    if series_id not in ids:
        raise ValueError(f"M4 series '{series_id}' was not found.")
    row = matrix.iloc[ids.index(series_id)].dropna().reset_index(drop=True)
    canonical = pd.DataFrame(
        {
            "date": _sequential_periods(len(row)),
            "series_id": str(series_id),
            "target": row,
        }
    )
    return canonical, {"format": "period-matrix", "period_index_is_sequential": True}


def _m5_sales_header(file_bytes: bytes, filename: str) -> list[str]:
    frame = _read_csv_bytes(file_bytes, filename, nrows=0)
    return [str(column) for column in frame.columns]


def m5_series_options(file_bytes: bytes, filename: str) -> pd.DataFrame:
    columns = _m5_sales_header(file_bytes, filename)
    metadata_columns = [column for column in M5_METADATA_COLUMNS if column in columns]
    if not metadata_columns:
        raise ValueError("M5 sales data must include id/item/store metadata columns and d_1, d_2, ... sales columns.")
    metadata = _read_csv_bytes(file_bytes, filename, usecols=metadata_columns, dtype=str)
    if "id" not in metadata.columns:
        parts = [column for column in ["item_id", "store_id"] if column in metadata.columns]
        if len(parts) < 2:
            raise ValueError("M5 sales data needs an id column or both item_id and store_id columns.")
        metadata["id"] = metadata[parts].fillna("").agg("_".join, axis=1)
    return metadata.drop_duplicates("id").reset_index(drop=True)


def load_m5_series(
    sales_bytes: bytes,
    sales_filename: str,
    calendar_bytes: bytes,
    calendar_filename: str,
    prices_bytes: bytes | None = None,
    prices_filename: str | None = None,
    series_id: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Read only one M5 sales row, then map d_n to calendar dates."""
    options = m5_series_options(sales_bytes, sales_filename)
    chosen = str(series_id or options.iloc[0]["id"])
    selected_meta = options[options["id"].astype(str) == chosen]
    if selected_meta.empty:
        raise ValueError(f"M5 series '{chosen}' was not found.")
    sales_columns = _m5_sales_header(sales_bytes, sales_filename)
    day_columns = [column for column in sales_columns if re.fullmatch(r"d_\d+", column)]
    if not day_columns:
        raise ValueError("M5 sales data does not contain d_1, d_2, ... day columns.")
    use_columns = ["id"] + day_columns if "id" in sales_columns else day_columns
    row = None
    if "id" in use_columns:
        for chunk in _read_csv_bytes(sales_bytes, sales_filename, usecols=use_columns, chunksize=2000):
            matches = chunk[chunk["id"].astype(str) == chosen]
            if not matches.empty:
                row = matches.iloc[[0]]
                break
    else:
        row = _read_csv_bytes(sales_bytes, sales_filename, usecols=use_columns).iloc[[selected_meta.index[0]]]
    if row is None or row.empty:
        raise ValueError(f"M5 sales row '{chosen}' was not found.")

    calendar = _read_csv_bytes(calendar_bytes, calendar_filename, usecols=["d", "date"])
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
    day_to_date = calendar.dropna(subset=["date"]).set_index("d")["date"]
    values = row.iloc[0][day_columns]
    canonical = pd.DataFrame(
        {
            "date": [day_to_date.get(day) for day in day_columns],
            "series_id": chosen,
            "target": pd.to_numeric(values.to_numpy(), errors="coerce"),
        }
    )
    info = {
        "format": "m5-sales-calendar",
        "item_id": str(selected_meta.iloc[0].get("item_id", "")),
        "store_id": str(selected_meta.iloc[0].get("store_id", "")),
        "state_id": str(selected_meta.iloc[0].get("state_id", "")),
        "prices_loaded": False,
    }
    if prices_bytes is not None and prices_filename is not None:
        price_columns = _m5_sales_header(prices_bytes, prices_filename)
        needed = [column for column in ["item_id", "store_id", "wm_yr_wk", "sell_price"] if column in price_columns]
        if "sell_price" not in needed:
            raise ValueError("The uploaded M5 prices file must contain sell_price.")
        _read_csv_bytes(prices_bytes, prices_filename, usecols=needed, nrows=1)
        info["prices_loaded"] = True
    return canonical, info