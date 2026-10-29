# Adaptive Forecasting Lab

An explainable Streamlit dashboard for comparing leakage-safe engineered feature sets on time-series benchmark data.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string
- `streamlit run app.py --server.port 5000 --server.address 0.0.0.0` — run the forecasting dashboard

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `app.py` — Streamlit navigation and dashboard presentation
- `src/` — data loading, preprocessing, EDA, feature engineering, selection, forecasting, metrics, and charts
- `config.json` — dataset filename patterns and lightweight experiment settings
- `data/` — user-provided official benchmark files
- `docs/` — project overview, methodology, and viva explanations

## Architecture decisions

- The app does not ship a fabricated benchmark or demo dataset; absent files produce an explicit upload message.
- Feature selection happens on the chronological training split only.
- Rolling features shift the input series before calculating statistics to avoid target leakage.
- The final comparison uses one Random Forest model plus a previous-observation baseline to stay explainable and lightweight.

## Product

The dashboard discovers CSV/XLSX benchmark files from `data/`, supports NN5 workbook metadata rows, lets users inspect EDA and engineered features, ranks and selects features on training data, runs a chronological forecasting experiment, and summarizes actual metrics.

## User preferences

Keep the project at an explainable third-year B.Tech level with minimal dependencies and no external APIs.

## Gotchas

- Benchmark files must be supplied by the user; the app intentionally refuses to invent benchmark results.
- Use `requirements.txt` to install the Python dependencies before running Streamlit.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
