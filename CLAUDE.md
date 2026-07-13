# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Purpose

Preflight is a single-page Dash app for quick dataset triage: upload a CSV/Parquet, check data health and missingness, set feature types, explore EDA/correlations, and train a baseline scikit-learn classifier — deployed at preflight.kooexperience.com.

## Tech Stack

- **App**: Dash 3.4 (Flask under the hood) + Dash Bootstrap Components (Darkly theme)
- **Data/ML**: pandas, numpy, scikit-learn (LogReg, RandomForest, KNN, SVM, Dummy baselines via `Pipeline` + `ColumnTransformer`)
- **Charts**: Plotly Express / Graph Objects
- **Server**: gunicorn (`wsgi:server`), Flask sessions for per-user dataset caching
- **Deploy**: Docker (`python:3.12-slim`) on Railway, driven by `railway.toml`

## Commands

Install (from repo root):
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run locally:
```bash
python app.py        # http://localhost:8050
```

Run with Docker:
```bash
docker build -t preflight:latest .
docker run --rm -p 8050:8050 preflight:latest
```

Tests (standalone scripts with `test_*` functions, runnable directly or via pytest):
```bash
python test_preservation.py
python test_memory_bug_exploration.py
```

## Architecture

Everything lives in one file, `app.py` (~2200 lines), structured top-to-bottom as:

1. **Config/env** — upload/row/column caps, rate limits, auth flags (`PREFLIGHT_*` env vars)
2. **Utilities** — dataset serialization to gzip'd JSON on disk keyed by session token (`save_df_for_session` / `load_df_for_session_token`), rate limiting, upload parsing, missing-token normalization, column typing heuristics (`auto_type_columns`), `ColumnTransformer` builder (`build_preprocessor`)
3. **Plot helpers** — confusion matrix, ROC/PR curves, missingness heatmap
4. **App + security** — Dash app init, Flask `before_request`/`after_request` guards (Basic Auth, CSP/security headers, same-origin POST check)
5. **Layout** — six `dcc.Tab` panes (Guide, Data Health, Missingness, Data Types, EDA, Model), each a `dbc.Card` built as a module-level `html.Div`
6. **Callbacks** (bulk of the file) — one section per pane, wiring uploads → dataset token → health profile → target selection → feature typing → EDA charts/correlation → model training/metrics

State is not held in server memory: the working dataframe is gzip-cached under `PREFLIGHT_DATA_DIR` (default `/tmp/preflight-data`) per Flask session, and only an opaque token travels through `dcc.Store`.

## Key Files

- `app.py` — entire application (layout, callbacks, data/ML logic)
- `wsgi.py` — exposes `server` (Flask) for gunicorn
- `requirements.txt` — pinned dependencies
- `Dockerfile` — production image, runs gunicorn with 2 workers/4 threads
- `railway.toml` — Railway build/deploy config (Dockerfile builder, healthcheck on `/`)
- `assets/dark-fix.css` — Dash auto-loaded CSS overrides for the dark theme
- `test_preservation.py`, `test_memory_bug_exploration.py` — behavior/memory regression checks for the data pipeline
