# Dataset-Adaptive Feature Engineering and Selection for Time-Series Forecasting

A ready-to-run B.Tech third-year Data Science mini-project implemented as a lightweight Streamlit dashboard.

## Project objective

The project studies whether automatically engineered and selected time-series features can improve or maintain forecasting performance compared with basic features or all engineered features. It focuses on an understandable experimental workflow rather than inventing a new forecasting algorithm.

The dashboard answers:

1. How many candidate features were generated?
2. How many features were selected?
3. Which features were selected and which were most important?
4. Did selected features improve or maintain forecasting performance?
5. How does feature selection behave across different datasets and series?

## Research contribution

The contribution is a dataset/series-adaptive experimental framework for analysing feature usefulness across benchmark time-series datasets. It is not a claim that the individual features, Random Forest, or correlation filtering are novel.

## Supported datasets

- **NN5** — common NN5 competition CSV/XLSX files, including the `NN5 COMPLETE Data` sheet and metadata rows before observations.
- **M4** — dated/long CSV/XLSX files and common period-matrix training files.
- **M5** — the related `sales_train_validation.csv`, `calendar.csv`, and optional `sell_prices.csv` files.

The earlier wide-table loader remains available for other dated CSV/XLSX benchmark files such as ETTh1 and Exchange-Rate through Auto-detect.

The app accepts CSV and XLSX/XLSM files. It does not ship a fabricated benchmark dataset. If no official benchmark file is available, the dashboard clearly asks you to upload one instead of showing invented metrics.

## Architecture

```text
Dataset
  -> validation and preprocessing
  -> exploratory data analysis
  -> leakage-safe feature engineering
  -> training-only feature selection
  -> Random Forest forecasting
  -> MAE / RMSE / sMAPE evaluation
  -> visual report and summary
```

### Project structure

```text
app.py                    Streamlit interface and navigation
config.json               Dataset filename patterns and experiment settings
requirements.txt          Minimal Python dependencies
src/data_loader.py        CSV/XLSX loading and NN5 handling
src/dataset_adapters.py   Common M4 and M5 adapters
src/preprocessing.py      Validation and missing-value preparation
src/eda.py                Statistics and temporal summaries
src/feature_engineering.py Leakage-safe feature creation
src/feature_selection.py  Correlation filtering and RF importance
src/forecasting.py        Chronological experiment runner
src/evaluation.py         MAE, RMSE, and sMAPE
src/visualization.py      Matplotlib chart helpers
data/                     User-provided benchmark files
results/                  Reserved for exported results
docs/                     Methodology and viva support
```

## Setup and run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the dashboard:

```bash
streamlit run app.py
```

For the configured Replit preview, this is equivalent to:

```bash
streamlit run app.py --server.port 5000 --server.address 0.0.0.0 --server.headless true
```

## How to add datasets

1. Download an official benchmark file.
2. Place it in `data/`, or upload it from the sidebar.
3. Choose **NN5**, **M4**, or **M5** from the benchmark selector.
4. Choose one series in the sidebar.

For M5, select **M5**, upload the sales and calendar files, and optionally upload the prices file. The app then shows the available item/store series IDs. Select one before opening EDA or Forecasting.

Filename patterns and defaults live in `config.json`, so adding another accepted filename does not require changing the main app. Uploaded files can also be selected directly from the sidebar.

## Features

The application generates:

- **Lags:** `lag_1`, `lag_7`, `lag_14`, `lag_28`
- **Rolling statistics:** seven- and fourteen-observation means and standard deviations
- **Calendar:** day of week, month, ISO week of year, and weekend flag
- **Changes:** one-period change, seven-period change, and seven-period growth

Rolling features shift the series before calculating their windows. This ensures the target observation cannot enter its own feature and avoids leakage.

## Feature selection

Selection is performed on the training portion only:

1. Highly correlated features are removed using the configured threshold.
2. A lightweight Random Forest Regressor calculates model-based feature importance.
3. The remaining features are ranked.
4. The top configured features become the selected engineered set.

The selected features and their importance values are visible in the dashboard.

## Forecasting and evaluation

The experiment uses a chronological split: the first 80% of complete feature rows are training data and the final 20% are test data. No random shuffling is used.

It compares:

- **Naive:** previous observation forecast
- **Basic:** a small set of lag, rolling, and calendar features
- **All Engineered:** all 15 generated features
- **Selected Engineered:** the training-only ranked subset

Metrics:

- **MAE:** average absolute error in target units
- **RMSE:** square-root average squared error, which emphasizes large errors
- **sMAPE:** symmetric percentage error, displayed as a percentage and safe when values are zero

## Limitations and future scope

This is intentionally a compact third-year project. It uses one main model, a one-step-ahead supervised setup, a simple holdout split, and a small feature vocabulary. It does not claim universal superiority.

Possible future work includes rolling-origin evaluation, additional classical forecasting models, richer frequency-aware features, hyperparameter tuning, and statistical significance testing across many series.

## Viva support

See:

- `docs/METHODOLOGY.md` for the end-to-end method.
- `docs/VIVA_GUIDE.md` for concise answers to common viva questions.
- `docs/PROJECT_OVERVIEW.md` for the problem statement and contribution.