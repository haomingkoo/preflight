# CLAUDE.md — preflight

Single-file Dash app (`app.py`, served by `wsgi:server`) live at preflight.kooexperience.com.

## Run and test

- `pip install -r requirements.txt` (Python 3.12; pytest is in requirements)
- `python app.py` serves on http://localhost:8050
- `pytest -q` runs `tests/` and the root `test_*.py` files (about 2 minutes). There is no CI, so run it before every PR.

## Deploy

- Railway project `serene-victory` auto-deploys every push to `main`, docs-only included, from the `Dockerfile` (`railway.toml`, healthcheck `/`).
- A redeploy wipes uploaded datasets. They live in `PREFLIGHT_DATA_DIR` (`/tmp/preflight-data`), not in a volume.

## Gotchas

- Gunicorn runs 2 workers. Keep per-user state in the on-disk session cache (`save_df_for_session`), never in process memory.
- Auth defaults on whenever `RAILWAY_ENVIRONMENT` is set. Startup fails closed if `PREFLIGHT_AUTH_USER`/`PREFLIGHT_AUTH_PASSWORD` are missing, unless `PREFLIGHT_REQUIRE_AUTH=0`.
- `/privacy` and `/terms` (`legal.py`) must stay outside Basic Auth (`PUBLIC_PATHS`). They describe what the code does, so update them when data handling changes.
- `pyarrow` is pinned to 25.0.0 because 26 needs numpy 2 and numpy is pinned to 1.26.4. Bump them together.
- No CDNs. Bootstrap Darkly and Inter are vendored in `assets/`. A test fails if the CSP or the CSS pulls fonts or styles from outside.

## Safety rules (enforced by `tests/test_privacy.py`)

- Never log dataset contents, exception messages or tracebacks. Use `log_error()`, which records only the exception type and a request id.
- Uploads require the consent checkbox server-side in `on_upload`. Do not drop that check.
- Uploaded data must stay deletable: "Delete my data now" plus the TTL cleanup thread (`PREFLIGHT_DATA_RETENTION_SECONDS`, default 1800).
