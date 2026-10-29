# Project overview

## Title

**Dataset-Adaptive Feature Engineering and Selection for Time-Series Forecasting**

## Problem

Time-series datasets can benefit from different representations. A large engineered feature set may contain redundant or unhelpful variables, while an overly small set may miss useful temporal signals. This project provides a practical framework for comparing those choices on benchmark series.

## Why multiple benchmarks?

NN5, M4, and M5 represent different forecasting settings and data formats. Using several benchmarks tests whether feature usefulness is consistent or changes with the dataset and selected series, while the downstream methodology remains the same.

## Common adapter architecture

Each input is converted to `date | series_id | target` before it reaches the shared pipeline. The NN5 loader handles workbook metadata, the M4 adapter handles dated/long files and period matrices, and the M5 adapter joins selected sales values to calendar dates. M5 prices are read when provided, but the current target is sales quantity.

M5 is multi-file because its day labels, retail sales quantities, and optional prices are stored separately. The app reads M5 metadata for selection and processes one item/store row instead of materializing every series.

## Contribution

The contribution is an explainable, dataset/series-adaptive experimental framework. It is not a claim that Random Forest feature selection or the individual features are novel.

## Outputs

For each chosen series, the dashboard reports dataset diagnostics, generated features, training-only feature importance, the selected subset, predictions, MAE, RMSE, and sMAPE.