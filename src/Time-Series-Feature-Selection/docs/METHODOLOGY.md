# Methodology

## 1. Dataset adapters

NN5, M4, and M5 inputs are converted to the same internal representation: `date | series_id | target`. NN5 uses the existing workbook-aware loader. M4 accepts long/dated tables and common one-row-per-series period matrices. M5 combines the selected `sales_train_validation.csv` row with actual dates from `calendar.csv`; `sell_prices.csv` is optional for the current sales-quantity target.

## 2. EDA

The app reports dimensions, series count, date range, missing values, descriptive statistics, distribution, observed values over time, rolling means, weekday patterns where possible, and engineered-feature correlations.

## 3. Preprocessing

Dates are parsed and sorted. Duplicate dates are reduced to one row. Numeric series are coerced to numbers. Missing values are counted before preprocessing, then time-interpolated with forward/backward filling for a usable single-series analysis.

## 4. Feature engineering

For a target at time `t`, lag and rolling features use only observations before `t`. The feature set includes lags, rolling means and standard deviations, calendar fields, changes, and seven-period growth.

## 5. Feature selection

The chronological training portion is used to remove highly correlated features. A Random Forest Regressor then ranks the retained features by model-based importance. The top ranked features form the selected engineered set. The test portion is never used for this step.

## 6. Forecasting

The main model is a lightweight Random Forest Regressor. The previous observation is included as a naive baseline. Three model feature sets are compared: basic, all engineered, and selected engineered.

## 7. Evaluation

The data is split chronologically, with the final 20% held out by default. MAE and RMSE are reported in the target's units. sMAPE is reported as a percentage and safely handles zero values.

## 8. Interpretation

Selected features are considered useful when they reduce the feature space while improving or maintaining test performance. Conclusions must be made per dataset/series and based on the displayed metrics.