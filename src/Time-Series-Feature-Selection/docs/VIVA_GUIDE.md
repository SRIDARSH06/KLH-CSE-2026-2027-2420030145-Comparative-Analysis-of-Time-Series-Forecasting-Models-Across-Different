# Viva guide

## What is the project?

It is an explainable framework that engineers time-series features, selects a useful subset using only training data, and compares forecasting performance against simpler feature sets.

## Why time-series forecasting?

Forecasting supports planning for demand, energy, finance, and operations where future values depend on patterns in earlier observations.

## Why feature engineering?

Raw observations do not directly expose all temporal patterns. Lags, rolling statistics, calendar variables, and changes give a traditional model useful signals.

## What is a lag feature?

A lag is an earlier observation used as a predictor, such as the value one or seven observations before the target.

## What is a rolling feature?

A rolling feature summarizes a moving window of earlier observations, such as a seven-observation mean. This project shifts the values first so the current target cannot enter its own rolling statistic.

## Why feature selection?

Selection can remove redundant or weak predictors, reduce computation, and make the model easier to explain.

## What makes the approach dataset-adaptive?

Feature importance and the final subset are learned separately for the selected series from its training data, rather than using one fixed subset for every dataset.

## Why use benchmark datasets?

Benchmarks give recognizable, reusable evaluation settings and make comparisons easier to reproduce.

## Why use multiple benchmarks?

NN5, M4, and M5 have different domains, frequencies, sizes, and file structures. Testing all three helps show whether useful engineered features are dataset- or series-dependent.

## Why NN5?

NN5 is a classic demand forecasting benchmark with multiple related daily cash-machine series, making it useful for studying series-to-series feature differences.

## What is M4?

M4 is a large forecasting benchmark containing series from several frequencies and domains. It may be provided as long/dated data or as rows of ordered period values, so the adapter standardizes either form.

## Why does M5 require multiple files?

M5 stores sales quantities, calendar dates, and prices in separate related files. The application joins the selected sales row's `d_1`, `d_2`, ... columns to the matching `d` and `date` values from the calendar file.

## How do you select an M5 series?

After uploading the sales and calendar files, choose the combined `id` shown in the sidebar, such as `FOODS_1_001_CA_1`. This represents an item/store series; the metadata also contains department, category, and state fields.

## What is the common adapter idea?

Each benchmark-specific adapter converts its input to date/period, series ID, and target columns. The same EDA, preprocessing, feature engineering, selection, model, and evaluation code then runs for every benchmark.

## Why use Random Forest?

It handles nonlinear relationships, works with mixed feature types, offers feature importance, and is simpler and lighter than deep learning for this project level.

## Why not use deep learning?

The project studies feature usefulness, not neural architecture design. Deep learning would add resource and explanation costs without being necessary for the research question.

## What is data leakage?

Leakage occurs when information unavailable at prediction time enters training or features. It makes evaluation look better than real forecasting.

## Why chronological train/test splitting?

Forecasting predicts the future, so earlier observations must train the model and later observations must test it. Random shuffling breaks that time order.

## What are MAE, RMSE, and sMAPE?

MAE is average absolute error. RMSE is the square root of average squared error and penalizes larger errors more. sMAPE scales absolute error by the magnitudes of actual and predicted values and is shown as a percentage.

## What is the research gap?

Many demonstrations focus on a single model or a fixed feature list. This project focuses on a transparent comparison of feature sets and how their usefulness changes by dataset and series.

## What is the project's contribution?

An accessible, series-adaptive experimental workflow that connects EDA, leakage-safe engineering, training-only selection, forecasting, and evaluation.

## What are the limitations?

It uses one main model, one-step-ahead supervised features, a simple chronological holdout, and a small engineered feature vocabulary. It does not claim universal superiority or replace careful benchmark-specific tuning.