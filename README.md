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

Security defaults:

- On Railway, HTTP Basic Auth is required by default.
- Set `PREFLIGHT_AUTH_USER`, `PREFLIGHT_AUTH_PASSWORD`, and `PREFLIGHT_SECRET_KEY` in Railway variables.
- If auth credentials are missing while auth is required, app startup fails closed.

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

Recommended secured run:

```bash
docker run --rm -p 8050:8050 \
  -e PREFLIGHT_REQUIRE_AUTH=1 \
  -e PREFLIGHT_AUTH_USER=admin \
  -e PREFLIGHT_AUTH_PASSWORD='change-me' \
  -e PREFLIGHT_SECRET_KEY='replace-with-random-long-string' \
  preflight:latest
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

## Security configuration

Environment variables used by the app:

- `PREFLIGHT_SECRET_KEY`: Flask session signing key (required in deployment environments)
- `PREFLIGHT_AUTH_USER`: HTTP Basic Auth username
- `PREFLIGHT_AUTH_PASSWORD`: HTTP Basic Auth password
- `PREFLIGHT_REQUIRE_AUTH`: enable/disable auth (`1`/`0`), defaults to `1` on Railway
- `PREFLIGHT_ENFORCE_POST_ORIGIN`: enforce same-origin on Dash POST callbacks (`1` by default)
- `PREFLIGHT_MAX_COLUMNS`: upload column cap (default `2000`)
- `PREFLIGHT_MAX_TOTAL_CELLS`: upload cell cap (default `5000000`)
- `PREFLIGHT_UPLOAD_RATE_LIMIT_PER_MIN`: upload requests per minute per IP (default `8`)
- `PREFLIGHT_TRAIN_RATE_LIMIT_PER_MIN`: training requests per minute per IP (default `12`)
- `PREFLIGHT_MAX_TRAIN_ROWS`: server-side train row cap (default `50000`)
- `PREFLIGHT_MAX_TRAIN_ROWS_SVM`: tighter train row cap for SVM (default `20000`)
- `PREFLIGHT_MAX_TRAIN_FEATURES`: maximum selected features for training (default `500`)
