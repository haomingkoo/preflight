# Preflight

Preflight is a Dash-based EDA and baseline classification workbench for quick dataset triage and model sanity checks.

## Current Railway deployment (dev)

Live URL:

`https://preflight-production-7851.up.railway.app/`

Deployment details in this repo:

- Build: Dockerfile (`[build] builder = "dockerfile"` in `railway.toml`)
- App server: `gunicorn wsgi:server`
- Bind: `0.0.0.0:${PORT:-8050}`
- Health check path: `/`
- Restart policy: `on_failure`

## What the app does

- Upload CSV or Parquet files
- Data health summary (missingness, uniques, duplicates)
- Missingness matrix and top missing columns
- Feature typing suggestions with manual overrides
- Per-feature EDA charts
- Correlation heatmap and top correlated pair plots
- Baseline classification models with CV metrics and confusion matrix

## Run locally

Prerequisites:

- Python 3.12+
- `pip`

From the `preflight/` folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open:

`http://localhost:8050`

## Run with Docker

From the `preflight/` folder:

```bash
docker build -t preflight:latest .
docker run --rm -p 8050:8050 preflight:latest
```

Open:

`http://localhost:8050`

## Deploy to Railway (dev)

This project is already configured for Railway via `railway.toml` and `Dockerfile`.

```bash
railway up
```

Useful commands:

```bash
railway logs
railway status
```

Note:

- Railway provides `PORT` at runtime; local default is `8050`.
