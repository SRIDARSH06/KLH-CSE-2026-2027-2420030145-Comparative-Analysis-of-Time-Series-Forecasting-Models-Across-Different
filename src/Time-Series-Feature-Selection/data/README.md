# Data folder

Place official benchmark files here without changing the application code:

- `nn5.xlsx` or `nn5.csv`
- `m4.csv`, `monthly-train.csv`, or another M4 CSV/XLSX file
- For M5, use the standard names `sales_train_validation.csv`, `calendar.csv`, and optionally `sell_prices.csv`

The loader looks at `config.json` to match filenames. It accepts wide tables with one date/time column and one or more numeric series. NN5 workbooks may use the common `NN5 COMPLETE Data` sheet and may contain instruction rows before the observations.

M4 can be uploaded as a dated/long table or as a common period matrix with one row per series. For a period matrix that has no real dates, the adapter preserves the ordered periods with a labelled sequential index.

M5 must be supplied as related files through the sidebar. The app reads the sales metadata to populate the series selector, maps `d_1`, `d_2`, ... to dates from `calendar.csv`, and reads only the chosen sales row. Prices are optional for the sales-quantity experiment.

The repository intentionally does not ship a fabricated benchmark or demo dataset. If no official file is available, the app shows an upload message instead of presenting invented results.