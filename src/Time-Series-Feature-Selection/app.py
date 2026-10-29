"""Streamlit dashboard for dataset-adaptive time-series feature experiments."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import discover_dataset_files, load_config, load_dataset, load_uploaded_dataset, validate_frame
from src.dataset_adapters import (
    canonical_to_wide,
    load_m4_series,
    load_m5_series,
    m4_series_options,
    m5_series_options,
)
from src.eda import descriptive_statistics, rolling_values, temporal_pattern
from src.feature_engineering import ALL_FEATURES, BASIC_FEATURES, FEATURE_DESCRIPTIONS, FEATURE_GROUPS, create_features
from src.forecasting import run_experiment
from src.preprocessing import validate_and_prepare
from src.visualization import (
    plot_correlation,
    plot_distribution,
    plot_forecasts,
    plot_importance,
    plot_rolling,
    plot_series,
    plot_temporal,
)


st.set_page_config(
    page_title="Adaptive Forecasting Lab",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def cached_config() -> dict:
    return load_config()


@st.cache_data(show_spinner=False)
def cached_path_dataset(path_string: str, dataset_id: str, sheet_name: str | None) -> pd.DataFrame:
    return load_dataset(Path(path_string), dataset_id, sheet_name)


@st.cache_data(show_spinner=False)
def cached_upload_dataset(file_bytes: bytes, filename: str, dataset_id: str, sheet_name: str | None) -> pd.DataFrame:
    return load_uploaded_dataset(file_bytes, filename, dataset_id, sheet_name)


@st.cache_data(show_spinner=False)
def cached_m4_options(file_bytes: bytes, filename: str) -> list[str]:
    return m4_series_options(file_bytes, filename)


@st.cache_data(show_spinner=False)
def cached_m4_frame(file_bytes: bytes, filename: str, series_id: str) -> pd.DataFrame:
    canonical, _ = load_m4_series(file_bytes, filename, series_id)
    return canonical_to_wide(canonical)


@st.cache_data(show_spinner=False)
def cached_m5_options(file_bytes: bytes, filename: str) -> pd.DataFrame:
    return m5_series_options(file_bytes, filename)


@st.cache_data(show_spinner=False)
def cached_m5_frame(
    sales_bytes: bytes,
    sales_filename: str,
    calendar_bytes: bytes,
    calendar_filename: str,
    prices_bytes: bytes | None,
    prices_filename: str | None,
    series_id: str,
) -> tuple[pd.DataFrame, dict]:
    canonical, info = load_m5_series(
        sales_bytes,
        sales_filename,
        calendar_bytes,
        calendar_filename,
        prices_bytes=prices_bytes,
        prices_filename=prices_filename,
        series_id=series_id,
    )
    return canonical_to_wide(canonical), info


def dataset_catalog(config: dict, discovered: dict[str, Path]) -> dict[str, dict]:
    catalog = {}
    for item in config.get("datasets", []):
        catalog[item["id"]] = {**item, "path": discovered.get(item["id"])}
    return catalog


def show_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def get_current_frame(catalog: dict[str, dict], choice: str, uploaded_file, interpretation: str) -> tuple[pd.DataFrame | None, str | None]:
    if uploaded_file is not None:
        dataset_id = interpretation if interpretation != "Auto-detect" else (
            "nn5" if "nn5" in uploaded_file.name.lower() else "auto"
        )
        sheet_name = catalog.get(dataset_id, {}).get("sheet_name")
        try:
            raw = cached_upload_dataset(uploaded_file.getvalue(), uploaded_file.name, dataset_id, sheet_name)
            return validate_frame(raw)[0], f"Uploaded file: {uploaded_file.name}"
        except Exception as error:
            st.error(f"Could not read the uploaded dataset: {error}")
            return None, None
    if choice == "Select a dataset":
        return None, None
    item = catalog[choice]
    if item["path"] is None:
        return None, None
    try:
        raw = cached_path_dataset(str(item["path"]), choice, item.get("sheet_name"))
        return validate_frame(raw)[0], f"{item['name']} file: {item['path'].name}"
    except Exception as error:
        st.error(f"Could not read the dataset: {error}")
        return None, None


def render_home(catalog: dict[str, dict]) -> None:
    show_header("Adaptive Forecasting Lab", "A transparent experiment for understanding whether useful engineered features vary by dataset and series.")
    st.info(
        "This project compares a previous-observation baseline, basic features, all engineered features, "
        "and a training-only selected feature set. Results are always calculated from the dataset you provide."
    )
    st.subheader("Research pipeline")
    st.markdown("**Dataset**  →  **Validation**  →  **EDA**  →  **Preprocessing**  →  **Feature engineering**  →  **Feature selection**  →  **Forecasting**  →  **Evaluation**")
    left, right = st.columns(2)
    with left:
        st.subheader("Project objectives")
        st.markdown(
            "- Make a compact, explainable forecasting workflow.\n"
            "- Generate leakage-safe lag, rolling, calendar, and change features.\n"
            "- Select features using training data only.\n"
            "- Compare performance and feature-space size across series."
        )
    with right:
        st.subheader("Supported benchmark datasets")
        for item in catalog.values():
            status = "Available in data/" if item["path"] else "Waiting for upload"
            st.markdown(f"**{item['name']}**  \n{item['description']}  \n`{status}`")
    st.subheader("Research questions")
    st.markdown(
        "Does a smaller, series-adaptive feature subset improve or maintain forecasting performance? "
        "How does the usefulness of engineered features change across benchmark datasets?"
    )


def render_dataset(frame: pd.DataFrame | None, source_label: str | None) -> None:
    show_header("Dataset", "Load an official benchmark file, inspect its structure, and choose a series for analysis.")
    if frame is None:
        st.warning("Upload an official benchmark dataset to run the final experiment.")
        st.markdown("Place a CSV or XLSX file in `data/`, or use the upload control in the sidebar. The loader supports wide tables with one date/time column and multiple numeric series.")
        return
    summary = validate_frame(frame)[1]
    first, second, third, fourth = st.columns(4)
    first.metric("Observations", f"{summary['rows']:,}")
    second.metric("Time series", summary["series_count"])
    third.metric("Missing values", f"{summary['missing_total']:,}")
    fourth.metric("Date range", f"{summary['start'].date()} – {summary['end'].date()}")
    st.caption(source_label)
    st.subheader("Preview")
    st.dataframe(frame.head(12), use_container_width=True, hide_index=True)
    st.subheader("Missing-value summary")
    missing = pd.DataFrame({"Series": list(summary["missing_by_series"]), "Missing values": list(summary["missing_by_series"].values())})
    st.dataframe(missing, use_container_width=True, hide_index=True)


def render_eda(series: pd.DataFrame | None, series_name: str) -> None:
    show_header("Exploratory analysis", f"Visual diagnostics for {series_name}.")
    if series is None:
        st.warning("Choose a dataset and series first.")
        return
    st.pyplot(plot_series(series, f"{series_name}: value over time"), clear_figure=True)
    left, right = st.columns(2)
    with left:
        st.pyplot(plot_distribution(series, f"{series_name}: value distribution"), clear_figure=True)
    with right:
        st.pyplot(plot_rolling(rolling_values(series), f"{series_name}: rolling means"), clear_figure=True)
    st.subheader("Descriptive statistics")
    st.dataframe(descriptive_statistics(series), use_container_width=True)
    pattern = temporal_pattern(series)
    if len(pattern) >= 5:
        st.pyplot(plot_temporal(pattern, f"{series_name}: day-of-week pattern"), clear_figure=True)

    features = create_features(series).dropna(subset=ALL_FEATURES)
    if len(features) > 10:
        st.pyplot(plot_correlation(features[ALL_FEATURES].corr(), "Engineered-feature correlation"), clear_figure=True)


def render_engineering(series: pd.DataFrame | None, series_name: str) -> None:
    show_header("Feature engineering", "Every row predicts its current target using only observations available before that target.")
    if series is None:
        st.warning("Choose a dataset and series first.")
        return
    features = create_features(series)
    st.metric("Candidate features generated", len(ALL_FEATURES))
    st.caption("Rows with the initial lag/rolling warm-up period are removed only when the forecasting experiment is run.")
    rows = []
    for group, names in FEATURE_GROUPS.items():
        for name in names:
            rows.append({"Group": group, "Feature": name, "Meaning": FEATURE_DESCRIPTIONS[name]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.subheader("Generated feature preview")
    st.dataframe(features.tail(8), use_container_width=True, hide_index=True)


def render_selection(series: pd.DataFrame | None, series_name: str, config: dict) -> None:
    show_header("Feature selection", "Redundant features are removed and Random Forest importance is calculated on chronological training data only.")
    if series is None:
        st.warning("Choose a dataset and series first.")
        return
    features = create_features(series)
    try:
        result = run_experiment(
            features,
            test_size=config["project"]["default_test_size"],
            correlation_threshold=config["project"]["correlation_threshold"],
            top_n=config["project"]["default_top_features"],
            n_estimators=config["project"]["random_forest_trees"],
        ).selection
    except ValueError as error:
        st.warning(str(error))
        return
    first, second, third = st.columns(3)
    first.metric("Candidate features", len(result.candidate_features))
    second.metric("After correlation filter", len(result.retained_after_correlation))
    third.metric("Selected features", len(result.selected_features))
    st.subheader("Importance ranking")
    st.pyplot(plot_importance(result.importance, f"{series_name}: training-only importance"), clear_figure=True)
    st.dataframe(result.importance.round(5), use_container_width=True, hide_index=True)
    st.subheader("Selected features")
    st.write(", ".join(result.selected_features))
    if result.redundant_features:
        st.caption(f"Removed as highly correlated: {', '.join(result.redundant_features)}")


def render_forecasting(series: pd.DataFrame | None, series_name: str, dataset_name: str, config: dict) -> None:
    show_header("Forecasting", "Run one chronological holdout experiment with a lightweight Random Forest and the previous-observation baseline.")
    if series is None:
        st.warning("Choose a dataset and series first.")
        return
    st.write(f"Selected series: **{series_name}**")
    st.caption("The default split uses the first 80% for training and the final 20% for testing. No random shuffling is used.")
    if st.button("Run forecasting experiment", type="primary"):
        try:
            with st.spinner("Training models on the selected series..."):
                result = run_experiment(
                    create_features(series),
                    test_size=config["project"]["default_test_size"],
                    correlation_threshold=config["project"]["correlation_threshold"],
                    top_n=config["project"]["default_top_features"],
                    n_estimators=config["project"]["random_forest_trees"],
                )
            st.session_state["experiment_result"] = result
            st.session_state["experiment_series"] = series_name
            st.session_state["experiment_dataset"] = dataset_name
            history = st.session_state.setdefault("benchmark_results", {})
            history[f"{dataset_name}|{series_name}"] = result.metrics.assign(
                Dataset=dataset_name,
                Series=series_name,
                **{"Features Used": result.metrics["Number of Features"]},
            )
            st.success("Experiment complete. Open Results to inspect the comparison.")
        except ValueError as error:
            st.error(str(error))


def render_results() -> None:
    show_header("Results", "Actual metrics and predictions from completed chronological experiments.")
    result = st.session_state.get("experiment_result")
    series_name = st.session_state.get("experiment_series", "selected series")
    if result is not None:
        metrics = result.metrics.copy()
        st.subheader(f"Latest experiment: {st.session_state.get('experiment_dataset', 'dataset')} / {series_name}")
        st.dataframe(metrics.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "sMAPE": "{:.2f}%"}), use_container_width=True, hide_index=True)
        st.pyplot(plot_forecasts(result.predictions, f"{series_name}: actual vs predicted"), clear_figure=True)
        all_count = int(metrics.loc[metrics["Feature Set"] == "All Engineered", "Number of Features"].iloc[0])
        selected_count = int(metrics.loc[metrics["Feature Set"] == "Selected Engineered", "Number of Features"].iloc[0])
        all_mae = float(metrics.loc[metrics["Feature Set"] == "All Engineered", "MAE"].iloc[0])
        selected_mae = float(metrics.loc[metrics["Feature Set"] == "Selected Engineered", "MAE"].iloc[0])
        reduction = (1 - selected_count / all_count) * 100 if all_count else 0
        mae_change = ((all_mae - selected_mae) / all_mae * 100) if all_mae else 0
        best_row = metrics.loc[metrics["MAE"].idxmin()]
        left, middle, right = st.columns(3)
        left.metric("Feature-space reduction", f"{reduction:.1f}%")
        middle.metric("Selected vs all MAE", f"{mae_change:+.1f}%")
        right.metric("Best by MAE", str(best_row["Feature Set"]))
        st.subheader("Project summary")
        direction = "lower" if selected_mae < all_mae else "higher" if selected_mae > all_mae else "the same"
        st.write(
            f"For **{series_name}**, feature selection reduced the engineered feature space from "
            f"{all_count} to {selected_count} features ({reduction:.1f}% reduction). The selected-feature "
            f"model achieved an MAE of {selected_mae:.4f}, compared with {all_mae:.4f} for the all-feature "
            f"model, so its MAE was {direction}. The best-performing row by MAE was **{best_row['Feature Set']}**."
        )
        st.caption(f"Training rows: {result.train_rows:,} | Test rows: {result.test_rows:,}")
    else:
        st.info("Run a forecasting experiment first.")

    st.subheader("Benchmark Comparison")
    history = st.session_state.get("benchmark_results", {})
    if history:
        comparison = pd.concat(history.values(), ignore_index=True)
        comparison = comparison[["Dataset", "Series", "Feature Set", "Features Used", "MAE", "RMSE", "sMAPE"]]
        st.dataframe(
            comparison.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "sMAPE": "{:.2f}%"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Run an experiment to populate actual benchmark comparison results.")
    for name in ["NN5", "M4", "M5"]:
        evaluated = any(str(key).startswith(f"{name}|") for key in history)
        if not evaluated:
            st.caption(f"{name}: Not evaluated — upload the benchmark dataset.")


def _benchmark_id(value: str) -> str:
    return {"NN5": "nn5", "M4": "m4", "M5": "m5"}.get(value, value)


def _infer_upload_type(filename: str) -> str:
    lowered = filename.lower()
    if "nn5" in lowered:
        return "nn5"
    if "m4" in lowered:
        return "m4"
    return "auto"


def main() -> None:
    config = cached_config()
    discovered = discover_dataset_files(config)
    catalog = dataset_catalog(config, discovered)
    st.sidebar.title("Adaptive Forecasting Lab")
    page = st.sidebar.radio("Navigate", ["Home", "Dataset", "EDA", "Feature Engineering", "Feature Selection", "Forecasting", "Results"])
    st.sidebar.divider()

    benchmark_options = ["Select a dataset", "NN5", "M4", "M5", "Auto-detect"]
    selected_benchmark = st.sidebar.selectbox("Benchmark dataset", benchmark_options)
    frame = None
    source_label = None
    dataset_name = selected_benchmark if selected_benchmark != "Select a dataset" else ""
    series_name = ""
    m5_info = None

    if selected_benchmark == "M5":
        st.sidebar.caption("M5 is a multi-file benchmark. Upload the matching files together.")
        sales_upload = st.sidebar.file_uploader("Sales data: sales_train_validation.csv", type=["csv"], key="m5_sales")
        calendar_upload = st.sidebar.file_uploader("Calendar: calendar.csv", type=["csv"], key="m5_calendar")
        prices_upload = st.sidebar.file_uploader("Prices: sell_prices.csv (optional)", type=["csv"], key="m5_prices")
        if sales_upload is not None and calendar_upload is not None:
            try:
                sales_bytes = sales_upload.getvalue()
                calendar_bytes = calendar_upload.getvalue()
                options = cached_m5_options(sales_bytes, sales_upload.name)
                series_ids = options["id"].astype(str).tolist()
                selected_m5_id = st.sidebar.selectbox(
                    "M5 item/store series",
                    series_ids,
                    format_func=lambda value: (
                        f"{value} · {options.loc[options['id'].astype(str) == value, 'item_id'].iloc[0]}"
                        if "item_id" in options.columns and not options.loc[options["id"].astype(str) == value].empty
                        else value
                    ),
                )
                frame, m5_info = cached_m5_frame(
                    sales_bytes,
                    sales_upload.name,
                    calendar_bytes,
                    calendar_upload.name,
                    prices_upload.getvalue() if prices_upload else None,
                    prices_upload.name if prices_upload else None,
                    selected_m5_id,
                )
                frame = validate_frame(frame)[0]
                series_name = selected_m5_id
                source_label = f"M5 files: {sales_upload.name} + {calendar_upload.name}"
                if prices_upload:
                    source_label += f" + {prices_upload.name}"
            except Exception as error:
                st.sidebar.error(f"Could not read the M5 files: {error}")
        else:
            st.sidebar.info("Upload both the sales and calendar files to choose an M5 series.")
    else:
        upload = st.sidebar.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx", "xlsm"], key="single_dataset")
        if upload is not None:
            interpretation = st.sidebar.selectbox("Interpret upload as", ["Auto-detect", "NN5", "M4", "M5"])
            dataset_id = _benchmark_id(interpretation)
            if dataset_id == "auto":
                dataset_id = _infer_upload_type(upload.name)
            try:
                if dataset_id == "m5":
                    st.sidebar.warning("Select M5 above and upload sales, calendar, and optional prices files separately.")
                elif dataset_id == "m4":
                    options = cached_m4_options(upload.getvalue(), upload.name)
                    series_name = st.sidebar.selectbox("M4 series", options)
                    frame = validate_frame(cached_m4_frame(upload.getvalue(), upload.name, series_name))[0]
                    dataset_name = "M4"
                    source_label = f"M4 upload: {upload.name}"
                else:
                    sheet_name = catalog.get(dataset_id, {}).get("sheet_name")
                    raw = cached_upload_dataset(upload.getvalue(), upload.name, dataset_id, sheet_name)
                    frame = validate_frame(raw)[0]
                    dataset_name = catalog.get(dataset_id, {}).get("name", "Auto-detected dataset")
                    source_label = f"Uploaded file: {upload.name}"
            except Exception as error:
                st.sidebar.error(f"Could not read the uploaded dataset: {error}")
        elif selected_benchmark in {"NN5", "M4"}:
            dataset_id = _benchmark_id(selected_benchmark)
            item = catalog.get(dataset_id, {})
            path = item.get("path")
            if path is not None:
                try:
                    if dataset_id == "m4":
                        options = cached_m4_options(path.read_bytes(), path.name)
                        series_name = st.sidebar.selectbox("M4 series", options)
                        frame = validate_frame(cached_m4_frame(path.read_bytes(), path.name, series_name))[0]
                    else:
                        raw = cached_path_dataset(str(path), dataset_id, item.get("sheet_name"))
                        frame = validate_frame(raw)[0]
                    dataset_name = item.get("name", selected_benchmark)
                    source_label = f"{dataset_name} file: {path.name}"
                except Exception as error:
                    st.sidebar.error(f"Could not read the dataset: {error}")

    series = None
    if frame is not None:
        names = [column for column in frame.columns if column != "date"]
        if not series_name or series_name not in names:
            series_name = st.sidebar.selectbox("Time series", names)
        try:
            series = validate_and_prepare(frame, series_name)[0]
        except ValueError as error:
            st.sidebar.error(str(error))
        if m5_info:
            st.sidebar.caption(
                f"Selected item {m5_info['item_id']} at {m5_info['store_id']} "
                f"({len(series) if series is not None else 0} observations)"
            )

    if page == "Home":
        render_home(catalog)
    elif page == "Dataset":
        render_dataset(frame, source_label)
    elif page == "EDA":
        render_eda(series, series_name)
    elif page == "Feature Engineering":
        render_engineering(series, series_name)
    elif page == "Feature Selection":
        render_selection(series, series_name, config)
    elif page == "Forecasting":
        render_forecasting(series, series_name, dataset_name, config)
    else:
        render_results()


if __name__ == "__main__":
    main()