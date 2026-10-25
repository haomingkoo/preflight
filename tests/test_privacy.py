from __future__ import annotations

import base64
import io
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app  # noqa: E402

SECRET_CELL = "alice@example.com"


def _csv_upload(text: str = f"name,score\n{SECRET_CELL},1\nbob,2\n") -> str:
    return "data:text/csv;base64," + base64.b64encode(text.encode()).decode()


@pytest.fixture()
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(app, "DATA_DIR", tmp_path)
    return tmp_path


@pytest.fixture()
def ctx():
    with app.server.test_request_context("/"):
        yield


def test_on_upload_without_consent_refuses_and_stores_nothing(data_dir: Path, ctx) -> None:
    result = app.on_upload(_csv_upload(), "people.csv", "", [])

    assert result[0] is None
    assert result[2] == ("error", app.CONSENT_REQUIRED_MESSAGE)
    assert list(data_dir.iterdir()) == []


def test_on_upload_with_consent_none_refuses(data_dir: Path, ctx) -> None:
    result = app.on_upload(_csv_upload(), "people.csv", "", None)

    assert result[2][0] == "error"


def test_on_upload_with_consent_stores_dataset(data_dir: Path, ctx) -> None:
    result = app.on_upload(_csv_upload(), "people.csv", "", [app.CONSENT_VALUE])

    assert result[0]
    assert result[2][0] == "success"
    assert len(list(data_dir.glob("*.json.gz"))) == 1


def test_on_upload_parses_parquet(data_dir: Path, ctx) -> None:
    buf = io.BytesIO()
    pd.DataFrame({"name": ["bob", "carol"], "score": [1, 2]}).to_parquet(buf)
    upload = "data:application/octet-stream;base64," + base64.b64encode(buf.getvalue()).decode()

    result = app.on_upload(upload, "people.parquet", "", [app.CONSENT_VALUE])

    assert result[2][0] == "success"
    assert len(list(data_dir.glob("*.json.gz"))) == 1


def test_delete_now_removes_session_file(data_dir: Path, ctx) -> None:
    app.on_upload(_csv_upload(), "people.csv", "", [app.CONSENT_VALUE])

    store, note, contents = app.on_delete_data(1)

    assert store is None
    assert contents is None
    assert note == ("success", "Your data was deleted from the server.")
    assert list(data_dir.iterdir()) == []
    assert "dataset_token" not in app.session


def test_delete_now_without_data_says_nothing_stored(data_dir: Path, ctx) -> None:
    _, note, _ = app.on_delete_data(1)

    assert note == ("success", "There was no data stored for this session.")


def test_log_error_omits_exception_message(caplog: pytest.LogCaptureFixture, ctx) -> None:
    caplog.set_level(logging.INFO, logger="preflight")

    app.log_error("Parsing failed", ValueError(f"could not convert string to float: '{SECRET_CELL}'"))

    text = caplog.text
    assert SECRET_CELL not in text
    assert "ValueError" in text
    assert "request_id=" in text


def test_unhandled_flask_errors_log_without_traceback(caplog: pytest.LogCaptureFixture, ctx) -> None:
    caplog.set_level(logging.INFO)
    try:
        raise KeyError(SECRET_CELL)
    except KeyError:
        app.server.log_exception(sys.exc_info())

    assert SECRET_CELL not in caplog.text
    assert "Traceback" not in caplog.text
    assert all(r.exc_info is None for r in caplog.records)


def test_training_failure_log_has_no_cell_values(caplog: pytest.LogCaptureFixture, ctx, monkeypatch) -> None:
    caplog.set_level(logging.INFO, logger="preflight")
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0] * 5, "y": [0, 1] * 10})

    def boom(*_a, **_k):
        raise ValueError(f"bad value {SECRET_CELL}")

    monkeypatch.setattr(app, "df_from_store", lambda _t: df)
    monkeypatch.setattr(app, "cross_val_score", boom)
    monkeypatch.setattr(app, "_allow_action", lambda *_a: True)
    app.train_model(1, "tok", "y", {"x": "numeric", "y": "categorical"}, "logreg", "f1_macro",
                    ["x"], 0.2, ["on"], 0.5, 0)

    assert SECRET_CELL not in caplog.text
    assert "Model training failed: ValueError" in caplog.text


def test_save_removes_temp_file_on_write_error(data_dir: Path, ctx, monkeypatch) -> None:
    def fail_replace(*_a):
        raise OSError("disk full")

    monkeypatch.setattr(app.os, "replace", fail_replace)

    with pytest.raises(OSError):
        app.save_df_for_session(pd.DataFrame({"a": [1]}))
    assert list(data_dir.iterdir()) == []


def test_forced_cleanup_deletes_expired_files(data_dir: Path) -> None:
    old = data_dir / "old.json.gz"
    old.write_bytes(b"x")
    past = time.time() - app.DATA_RETENTION_SECONDS - 10
    os.utime(old, (past, past))
    fresh = data_dir / "fresh.json.gz"
    fresh.write_bytes(b"x")

    app._cleanup_stale_datasets(force=True)

    assert not old.exists()
    assert fresh.exists()


def test_prune_rate_limit_buckets_drops_idle_ips() -> None:
    app.RATE_LIMIT_BUCKETS.clear()
    app.RATE_LIMIT_BUCKETS[("203.0.113.9", "upload")].append(time.time() - 120)
    app.RATE_LIMIT_BUCKETS[("203.0.113.10", "upload")].append(time.time())

    removed = app.prune_rate_limit_buckets()

    assert removed == 1
    assert list(app.RATE_LIMIT_BUCKETS) == [("203.0.113.10", "upload")]


def test_cleanup_thread_is_running() -> None:
    import threading

    assert any(t.name == "preflight-cleanup" for t in threading.enumerate())


@pytest.mark.parametrize("path", ["/privacy", "/terms"])
def test_legal_pages_render(path: str) -> None:
    resp = app.server.test_client().get(path)

    assert resp.status_code == 200
    assert b"privacy@kooexperience.com" in resp.data
    for marker in (b"LAWYER REVIEW", b"CONTACT EMAIL", b"owner to confirm", "\u2014".encode()):
        assert marker not in resp.data


def test_csp_has_no_font_or_style_cdns() -> None:
    assert "fonts.googleapis.com" not in app.CSP_POLICY
    assert "fonts.gstatic.com" not in app.CSP_POLICY
    assert "cdn.jsdelivr.net" not in app.CSP_POLICY


def test_vendored_css_has_no_remote_imports() -> None:
    assets = Path(app.__file__).parent / "assets"
    css = "".join(p.read_text() for p in assets.glob("*.css"))

    assert "@import" not in css
    assert "googleapis" not in css


@pytest.mark.parametrize("path", ["/", "/privacy", "/terms"])
def test_pages_share_footer_and_favicon(path: str) -> None:
    html = app.server.test_client().get(path).get_data(as_text=True)
    if path == "/":
        # The Dash shell renders the layout client-side; the footer lives in the layout.
        html += str(app.app.layout)
    assert "/assets/favicon.svg" in html
    assert "2026 Haoming Koo" in html
    assert "kooexperience.com" in html
    assert "github.com/haomingkoo/preflight" in html


def test_legal_pages_load_their_assets_without_auth() -> None:
    assert {"/assets/favicon.svg", "/assets/fonts/hanken-grotesk-latin-wght-normal.woff2"} <= app.PUBLIC_PATHS
