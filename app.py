"""
EDA + Classification Workbench
Run:
  python app.py

Hard fixes:
- Typing dropdown is clickable (DataTable css + overflow fixes)
- Missingness hover shows full column names (2D customdata aligned to z)
- DataTable pagination input/labels are white on dark theme
- Model features populate after Save Types
- EDA has Select all / Clear and optional scatter for 2 numeric features
"""

import base64
import binascii
from collections import defaultdict, deque
from datetime import timedelta
import gc
import gzip
import io
from dataclasses import dataclass
import logging
from pathlib import Path
import secrets
import threading
import textwrap
import time
from typing import Deque, Dict, List, Optional, Tuple
import os
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State, dash_table, callback_context, no_update
import dash_bootstrap_components as dbc
from dash.dash_table.Format import Format, Group, Scheme
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from flask import Response, g, request, session

import legal

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
)

# -----------------------
# Defaults
# -----------------------
MISSING_TOKENS_DEFAULT = ["?", "NA", "N/A", "null", "None", "nan", "NaN", ""]
RANDOM_STATE = 42

MAX_UPLOAD_MB = 50
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_ROWS_WARN = 300_000
MAX_COLUMNS = int(os.getenv("PREFLIGHT_MAX_COLUMNS", "2000"))
MAX_TOTAL_CELLS = int(os.getenv("PREFLIGHT_MAX_TOTAL_CELLS", "5000000"))
MAX_SERIALIZED_DATA_BYTES = int(os.getenv("PREFLIGHT_MAX_SERIALIZED_DATA_BYTES", "120000000"))
MAX_TRAIN_ROWS = int(os.getenv("PREFLIGHT_MAX_TRAIN_ROWS", "50000"))
MAX_TRAIN_ROWS_SVM = int(os.getenv("PREFLIGHT_MAX_TRAIN_ROWS_SVM", "20000"))
MAX_TRAIN_FEATURES = int(os.getenv("PREFLIGHT_MAX_TRAIN_FEATURES", "500"))
UPLOAD_RATE_LIMIT_PER_MIN = int(os.getenv("PREFLIGHT_UPLOAD_RATE_LIMIT_PER_MIN", "8"))
TRAIN_RATE_LIMIT_PER_MIN = int(os.getenv("PREFLIGHT_TRAIN_RATE_LIMIT_PER_MIN", "12"))
DATA_RETENTION_SECONDS = int(os.getenv("PREFLIGHT_DATA_RETENTION_SECONDS", "1800"))
DATA_DIR = Path(os.getenv("PREFLIGHT_DATA_DIR", "/tmp/preflight-data"))
CLEANUP_INTERVAL_SECONDS = int(os.getenv("PREFLIGHT_CLEANUP_INTERVAL_SECONDS", "60"))
RATE_LIMIT_WINDOW_SECONDS = 60
CONSENT_VALUE = "agree"
CONSENT_TEXT = (
    "I have the right to upload this data and it contains no personal data of others, "
    "or I have permission to process it."
)
CONSENT_REQUIRED_MESSAGE = "Tick the box to confirm you have the right to upload this data, then upload again."

LOGGER = logging.getLogger("preflight")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
RATE_LIMIT_LOCK = threading.Lock()
RATE_LIMIT_BUCKETS: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)
LAST_DATA_CLEANUP_EPOCH = 0.0

CV_METRICS = [
    ("F1 Macro", "f1_macro"),
    ("Accuracy", "accuracy"),
    ("Recall Macro", "recall_macro"),
]
BINARY_ONLY_METRICS = [
    ("ROC AUC (binary)", "roc_auc"),
    ("Average Precision (binary)", "average_precision"),
]


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_REQUIRE_AUTH = "1" if os.getenv("RAILWAY_ENVIRONMENT") else "0"
REQUIRE_AUTH = env_flag("PREFLIGHT_REQUIRE_AUTH", default=DEFAULT_REQUIRE_AUTH == "1")
AUTH_USER = os.getenv("PREFLIGHT_AUTH_USER", "")
AUTH_PASSWORD = os.getenv("PREFLIGHT_AUTH_PASSWORD", "")
ENFORCE_POST_ORIGIN = env_flag("PREFLIGHT_ENFORCE_POST_ORIGIN", default=True)
SESSION_COOKIE_SECURE = env_flag(
    "PREFLIGHT_SESSION_COOKIE_SECURE", default=bool(os.getenv("RAILWAY_ENVIRONMENT"))
)

CSP_POLICY = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://static.cloudflareinsights.com; "
    "style-src 'self' 'unsafe-inline'; "
    "font-src 'self' data:; "
    "img-src 'self' data: blob:; "
    "media-src 'self'; "
    "connect-src 'self' https://cloudflareinsights.com; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)

if REQUIRE_AUTH and (not AUTH_USER or not AUTH_PASSWORD):
    raise RuntimeError(
        "PREFLIGHT_REQUIRE_AUTH is enabled but PREFLIGHT_AUTH_USER/PREFLIGHT_AUTH_PASSWORD are missing."
    )

# -----------------------
# Utilities
# -----------------------
def _request_id() -> str:
    try:
        rid = g.get("request_id")
        if not rid:
            rid = secrets.token_hex(8)
            g.request_id = rid
        return rid
    except RuntimeError:  # outside a request context
        return "no-request"


def log_error(context: str, exc: BaseException) -> None:
    """Log only the exception type and request id. Messages can carry uploaded cell values."""
    LOGGER.error("%s: %s (request_id=%s)", context, type(exc).__name__, _request_id())


def log_memory_usage(operation: str, before_mb: Optional[float] = None) -> None:
    """
    Log memory usage for monitoring and troubleshooting.
    
    Args:
        operation: Name of the operation (e.g., 'upload', 'eda', 'training')
        before_mb: Optional memory usage before operation (for delta calculation)
    """
    try:
        import psutil
        process = psutil.Process()
        current_mb = process.memory_info().rss / (1024 * 1024)
        
        if before_mb is not None:
            delta = current_mb - before_mb
            LOGGER.info(f"Memory [{operation}]: {current_mb:.1f} MB (delta: {delta:+.1f} MB)")
        else:
            LOGGER.info(f"Memory [{operation}]: {current_mb:.1f} MB")
    except ImportError:
        # psutil not available, skip logging
        pass
    except Exception:
        # Don't let memory logging break the app
        pass


def shorten_label(s: str, max_len: int = 18) -> str:
    s = str(s)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def df_from_store(df_json):
    if not df_json:
        return None
    if isinstance(df_json, dict):  # orient="split" dict
        return pd.DataFrame(df_json["data"], columns=df_json["columns"], index=df_json["index"])
    if isinstance(df_json, str) and df_json.lstrip().startswith("{"):
        return pd.read_json(io.StringIO(df_json), orient="split")
    return load_df_for_session_token(df_json)


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(DATA_DIR, 0o700)
    except OSError:
        pass


def _client_ip() -> str:
    xff = (request.headers.get("X-Forwarded-For") or "").strip()
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _allow_action(action: str, limit_per_minute: int) -> bool:
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    key = (_client_ip(), action)
    with RATE_LIMIT_LOCK:
        bucket = RATE_LIMIT_BUCKETS[key]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= limit_per_minute:
            return False
        bucket.append(now)
    return True


def prune_rate_limit_buckets(now: Optional[float] = None) -> int:
    """Drop buckets with no hits inside the window so client IPs do not linger in memory."""
    cutoff = (now or time.time()) - RATE_LIMIT_WINDOW_SECONDS
    with RATE_LIMIT_LOCK:
        stale = [k for k, b in RATE_LIMIT_BUCKETS.items() if not b or b[-1] < cutoff]
        for k in stale:
            del RATE_LIMIT_BUCKETS[k]
    return len(stale)


def _ensure_session_id() -> str:
    sid = session.get("sid")
    if sid:
        return sid
    sid = secrets.token_urlsafe(24)
    session["sid"] = sid
    session.modified = True
    return sid


def _dataset_path_for_current_session() -> Path:
    sid = "".join(ch for ch in _ensure_session_id() if ch.isalnum() or ch in "-_")
    return DATA_DIR / f"{sid}.json.gz"


def _cleanup_stale_datasets(force: bool = False) -> None:
    global LAST_DATA_CLEANUP_EPOCH
    now = time.time()
    if not force and now - LAST_DATA_CLEANUP_EPOCH < 300:
        return
    LAST_DATA_CLEANUP_EPOCH = now
    cutoff = now - DATA_RETENTION_SECONDS
    deleted_count = 0
    for dataset_path in [*DATA_DIR.glob("*.json.gz"), *DATA_DIR.glob("*.json.tmp")]:
        try:
            if dataset_path.stat().st_mtime < cutoff:
                dataset_path.unlink(missing_ok=True)
                deleted_count += 1
        except OSError:
            continue
    
    # Trigger garbage collection to release memory from deleted sessions
    if deleted_count > 0:
        LOGGER.info(f"Cleaned up {deleted_count} stale dataset(s), triggering GC")
        gc.collect()


def _cleanup_loop() -> None:
    while True:
        time.sleep(CLEANUP_INTERVAL_SECONDS)
        try:
            _cleanup_stale_datasets(force=True)
            prune_rate_limit_buckets()
        except Exception as exc:  # keep the timer alive
            log_error("Background cleanup failed", exc)


def start_cleanup_thread() -> threading.Thread:
    thread = threading.Thread(target=_cleanup_loop, name="preflight-cleanup", daemon=True)
    thread.start()
    return thread


def delete_session_dataset() -> bool:
    """Remove the current session's dataset and its temp file. Returns True if a file was removed."""
    destination = _dataset_path_for_current_session()
    removed = False
    for path in (destination, destination.with_suffix(".tmp")):
        try:
            if path.exists():
                path.unlink()
                removed = True
        except OSError as exc:
            log_error("Failed to delete session dataset", exc)
    session.pop("dataset_token", None)
    session.modified = True
    return removed


def save_df_for_session(df: pd.DataFrame) -> str:
    payload = df.to_json(date_format="iso", orient="split")
    encoded = payload.encode("utf-8")
    if len(encoded) > MAX_SERIALIZED_DATA_BYTES:
        raise ValueError(
            "Dataset is too large after parsing. Reduce row/column count and try again."
        )
    destination = _dataset_path_for_current_session()
    temp_path = destination.with_suffix(".tmp")
    try:
        with gzip.open(temp_path, "wb") as f:
            f.write(encoded)
        try:
            os.chmod(temp_path, 0o600)
        except OSError:
            pass
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
    token = secrets.token_urlsafe(16)
    session["dataset_token"] = token
    session.modified = True
    return token


def load_df_for_session_token(token: str) -> Optional[pd.DataFrame]:
    if not token:
        return None
    expected = session.get("dataset_token")
    if expected and token != expected:
        return None
    dataset_path = _dataset_path_for_current_session()
    if not dataset_path.exists():
        return None
    try:
        with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
            payload = f.read()
        return pd.read_json(io.StringIO(payload), orient="split")
    except Exception as exc:
        log_error("Failed to load cached dataset for session", exc)
        return None


def _unauthorized_response() -> Response:
    return Response(
        "Authentication required.",
        status=401,
        headers={"WWW-Authenticate": 'Basic realm="Preflight"'},
    )


def _is_authorized_request() -> bool:
    auth = request.authorization
    if not auth:
        return False
    username_ok = secrets.compare_digest(auth.username or "", AUTH_USER)
    password_ok = secrets.compare_digest(auth.password or "", AUTH_PASSWORD)
    return username_ok and password_ok


def _is_same_origin(origin: str) -> bool:
    try:
        origin_url = urlparse(origin)
        host_url = urlparse(request.host_url)
    except ValueError:
        return False
    return origin_url.scheme in {"http", "https"} and origin_url.netloc == host_url.netloc


def enforce_dataframe_limits(df: pd.DataFrame) -> pd.DataFrame:
    rows, cols = df.shape
    total_cells = rows * cols
    if cols > MAX_COLUMNS:
        raise ValueError(
            f"Too many columns ({cols:,}). Maximum allowed is {MAX_COLUMNS:,}."
        )
    if total_cells > MAX_TOTAL_CELLS:
        raise ValueError(
            f"Dataset is too large ({total_cells:,} cells). Maximum allowed is {MAX_TOTAL_CELLS:,} cells."
        )
    return df

def apply_type_overrides(df: pd.DataFrame, feature_types: Optional[Dict[str, str]]) -> pd.DataFrame:
    """
    Apply semantic typing decisions (from store-feature-types) onto dataframe dtypes.
    This is mainly to prevent encoded categorical ints from leaking into numeric-only EDA/correlation.
    """
    if df is None or not feature_types:
        return df

    df2 = df.copy()
    for col, t in feature_types.items():
        if col not in df2.columns:
            continue

        if t in ("categorical", "high_card_categorical"):
            # force categorical semantics
            df2[col] = df2[col].astype("category")

        elif t == "numeric":
            df2[col] = pd.to_numeric(df2[col], errors="coerce")

        elif t == "ordinal":
            df2[col] = pd.to_numeric(df2[col], errors="coerce")
        # if t == "drop": do nothing here; filtering happens at selection time

    return df2

def parse_contents(contents: str, filename: str) -> pd.DataFrame:
    try:
        _, content_string = contents.split(",", 1)
    except ValueError as exc:
        raise ValueError("Invalid upload payload.") from exc
    approx_bytes = len(content_string) * 3 // 4
    if approx_bytes > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"File too large (~{approx_bytes // (1024 * 1024):.0f} MB). "
            f"Maximum upload size is {MAX_UPLOAD_MB} MB."
        )
    try:
        decoded = base64.b64decode(content_string, validate=True)
    except binascii.Error as exc:
        raise ValueError("Invalid upload payload.") from exc

    if filename.lower().endswith(".csv"):
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8", errors="ignore")), low_memory=False)
        return enforce_dataframe_limits(df)
    if filename.lower().endswith(".parquet"):
        df = pd.read_parquet(io.BytesIO(decoded))
        return enforce_dataframe_limits(df)
    raise ValueError("Unsupported file type. Upload a CSV or Parquet.")


def normalize_missing_tokens(df: pd.DataFrame, missing_tokens: List[str]) -> pd.DataFrame:
    df2 = df.copy()
    for col in df2.columns:
        if df2[col].dtype == object:
            df2[col] = df2[col].replace(missing_tokens, np.nan)
    return df2

def make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen = {}
    new_cols = []
    for c in df.columns:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}__dup{seen[c]}")
    df = df.copy()
    df.columns = new_cols
    return df

def basic_profile(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "missing_%": float(s.isna().mean() * 100),
                "n_unique": int(s.nunique(dropna=True)),
                "unique_ratio": float(s.nunique(dropna=True) / max(n, 1)),
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_%", "n_unique"], ascending=[False, False])


@dataclass
class TypingConfig:
    low_card_threshold: int = 20
    id_unique_ratio_threshold: float = 0.90
    treat_small_unique_int_as_cat: bool = True


def auto_type_columns(df: pd.DataFrame, target_col: str, cfg: TypingConfig) -> Dict[str, str]:
    n = len(df)
    types: Dict[str, str] = {}
    for col in df.columns:
        if col == target_col:
            continue

        s = df[col]
        nunique = s.nunique(dropna=True)
        unique_ratio = nunique / max(n, 1)

        if unique_ratio >= cfg.id_unique_ratio_threshold:
            types[col] = "drop"
            continue

        if pd.api.types.is_numeric_dtype(s):
            if cfg.treat_small_unique_int_as_cat and nunique <= cfg.low_card_threshold:
                types[col] = "categorical"
            else:
                types[col] = "numeric"
        else:
            if nunique <= cfg.low_card_threshold:
                types[col] = "categorical"
            else:
                types[col] = "high_card_categorical"
    return types


def build_preprocessor(
    feature_types: Dict[str, str],
    scale_numeric: bool,
    available_cols: List[str],
) -> Tuple[ColumnTransformer, List[str], List[str], List[str]]:
    available = set(available_cols)

    numeric_cols = [c for c, t in feature_types.items() if t == "numeric" and c in available]
    ordinal_cols = [c for c, t in feature_types.items() if t == "ordinal" and c in available]
    cat_cols = [
        c
        for c, t in feature_types.items()
        if t in ("categorical", "high_card_categorical") and c in available
    ]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=num_steps)

    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("ord", ordinal_transformer, ordinal_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric_cols, ordinal_cols, cat_cols


# -----------------------
# Plot helpers
# -----------------------
def fig_confusion_matrix(cm: np.ndarray, labels: List[str]) -> go.Figure:
    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=labels,
        labels={"x": "Predicted", "y": "Actual"},
        title="Confusion Matrix",
        template="preflight",
    )
    fig.update_layout(height=CHART_HEIGHT, margin=dict(l=40, r=40, t=60, b=40))
    return fig


def fig_roc_pr(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[go.Figure, go.Figure, float, float]:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(rec, prec)
    ap = average_precision_score(y_true, y_proba)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC AUC={roc_auc:.3f}"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
    fig_roc.update_layout(
        height=CHART_HEIGHT,
        title="ROC Curve",
        template="preflight",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"PR AUC={pr_auc:.3f}, AP={ap:.3f}"))
    fig_pr.update_layout(
        height=CHART_HEIGHT,
        title="Precision-Recall Curve",
        template="preflight",
        xaxis_title="Recall",
        yaxis_title="Precision",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig_roc, fig_pr, roc_auc, ap

def missingness_heatmap_figure(
    df: pd.DataFrame,
    sample_n: int,
    seed: int,
    col_mode: str,
    top_n: int,
) -> go.Figure:
    if df is None or len(df) == 0:
        return placeholder_fig("Upload a file to see where its values are missing.")

    seed = int(seed or 42)
    sample_n = int(max(50, min(int(sample_n or 5000), len(df))))

    miss_rate = df.isna().mean().sort_values(ascending=False)

    if col_mode == "top_missing":
        cols = miss_rate[miss_rate > 0].head(int(top_n)).index.tolist()
        if not cols:
            cols = df.columns.tolist()[: min(df.shape[1], int(top_n))]
    else:
        cols = df.columns.tolist()

    # df_s = df.sample(sample_n, random_state=seed)

    if sample_n >= len(df):
        # No sampling → preserve original order
        df_s = df
    else:
        # Sampling → reproducible randomness
        df_s = df.sample(sample_n, random_state=seed)

    # z shape: (rows, cols); 0=valid, 1=missing
    mat = df_s[cols].isna().astype(int).to_numpy()

    x_full = [str(c) for c in cols]
    x_tick = [shorten_label(c, 18) for c in x_full]

    y_vals = df_s.index.astype(str).tolist()

    fig = go.Figure(
        data=go.Heatmap(
            z=mat,
            x=x_full,                 # FULL names live here
            y=y_vals,
            hovertemplate="Row=%{y}<br>Column=%{x}<br>Missing=%{z}<extra></extra>",
            showscale=False,
            zmin=0,
            zmax=1,
            # Present cells stay pale; missing cells are the one dark mark.
            colorscale=[[0.0, "#e6ecec"], [1.0, "#0b5953"]],
            # xgap=1,   # vertical grid lines (between columns)
            # ygap=0,   # horizontal grid lines (between rows)
        )
    )

    # Show shortened labels while keeping x values as full names
    fig.update_xaxes(
        tickmode="array",
        tickvals=x_full,
        ticktext=x_tick,
        tickangle=45,
        automargin=True,
    )
    fig.update_yaxes(
        showticklabels=False,
        autorange="reversed"
    )

    fig.update_layout(
        template="preflight",
        title=f"Missing values by row ({sample_n:,} rows, {len(cols):,} columns, seed {seed})",
        margin=dict(l=40, r=40, t=60, b=120),
        height=520,
    )
    return fig

# -----------------------
# Chart theme: plotly_white tuned to the page tokens, used by every figure
# -----------------------
CHART_FONT = "Hanken Grotesk, system-ui, -apple-system, Segoe UI, sans-serif"
CHART_INK = "#3a4448"
CHART_GRID = "#e3e8e8"
# Explicit heights: dcc.Graph otherwise keeps the size of the previous figure (e.g. a short placeholder).
CHART_HEIGHT = 450
PLACEHOLDER_HEIGHT = 160
_theme = go.layout.Template(pio.templates["plotly_white"])
_theme.layout.update(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=CHART_FONT, color=CHART_INK, size=13),
    title=dict(font=dict(size=15, color="#141a1d"), x=0, xanchor="left", pad=dict(l=4)),
    # Teal first, then hues that stay distinct for common colour-vision deficiencies.
    colorway=["#0f766e", "#b45309", "#1d4ed8", "#be185d", "#4d7c0f", "#6d28d9", "#0e7490", "#9f1239"],
    hoverlabel=dict(bgcolor="#ffffff", bordercolor="#8a9494", font=dict(family=CHART_FONT, color="#141a1d")),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)
_theme.layout.xaxis.update(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID, linecolor="#8a9494")
_theme.layout.yaxis.update(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID, linecolor="#8a9494")
_theme.layout.colorscale.sequential = [[0.0, "#f0f7f6"], [0.5, "#5fb3a9"], [1.0, "#0b4f4a"]]
_theme.layout.colorscale.diverging = [[0.0, "#8c510a"], [0.5, "#f5f5f5"], [1.0, "#01665e"]]
pio.templates["preflight"] = _theme
pio.templates.default = "preflight"


def placeholder_fig(message: str) -> go.Figure:
    """Empty-state chart: a short sentence instead of a blank grid."""
    fig = go.Figure()
    fig.update_layout(
        height=PLACEHOLDER_HEIGHT,
        template="preflight",
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(text="<br>".join(textwrap.wrap(message, 34)), showarrow=False, font=dict(size=14, color="#56616a"),
                          xref="paper", yref="paper", x=0.5, y=0.5)],
    )
    return fig


# -----------------------
# App + CSS
# -----------------------
# Bootstrap and the theme CSS are vendored in assets/ so no CDN is contacted.
app = Dash(__name__)
server = app.server
app.title = "Preflight: check a dataset before you model it"
_ensure_data_dir()
start_cleanup_thread()

default_secret = None if os.getenv("RAILWAY_ENVIRONMENT") else secrets.token_hex(32)
secret_key = os.getenv("PREFLIGHT_SECRET_KEY", default_secret)
if not secret_key:
    raise RuntimeError(
        "PREFLIGHT_SECRET_KEY is required in deployment environments."
    )

server.secret_key = secret_key
server.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=SESSION_COOKIE_SECURE,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
)


# Legal pages and the two static files they load stay reachable without Basic Auth.
PUBLIC_PATHS = {"/privacy", "/terms", "/assets/favicon.svg", "/assets/fonts/hanken-grotesk-latin-wght-normal.woff2"}


def _log_exception_without_data(exc_info) -> None:
    """Replace Flask's traceback logging: tracebacks can echo uploaded cell values."""
    log_error(f"Unhandled error on {request.path}", exc_info[1])


server.log_exception = _log_exception_without_data


@server.before_request
def security_guard():
    g.request_id = secrets.token_hex(8)
    _ensure_session_id()
    _cleanup_stale_datasets()

    if REQUIRE_AUTH and request.path not in PUBLIC_PATHS and not _is_authorized_request():
        return _unauthorized_response()

    if request.method == "GET" and request.path not in APP_PAGES and not request.path.startswith(("/_", "/assets/")):
        return _not_found()

    if ENFORCE_POST_ORIGIN and request.method == "POST" and request.path.startswith("/_dash"):
        origin = request.headers.get("Origin")
        if origin and not _is_same_origin(origin):
            return Response("Forbidden origin.", status=403)

    return None


@server.after_request
def add_security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    resp.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    resp.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    resp.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
    resp.headers.setdefault("Content-Security-Policy", CSP_POLICY)
    if request.headers.get("X-Forwarded-Proto", "").lower() == "https" or request.is_secure:
        resp.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return resp

APP_PAGES = {"/", "/privacy", "/terms"}


@server.errorhandler(404)
def _not_found(_exc=None) -> Response:
    return Response(legal.NOT_FOUND_HTML, status=404, mimetype="text/html")


@server.route("/privacy")
def privacy_page() -> Response:
    return Response(legal.PRIVACY_HTML, mimetype="text/html")


@server.route("/terms")
def terms_page() -> Response:
    return Response(legal.TERMS_HTML, mimetype="text/html")


app.index_string = """
<!DOCTYPE html>
<html lang="en">
  <head>
    {%metas%}
    <title>{%title%}</title>
    <meta name="theme-color" content="#ffffff" />
    <meta name="description" content="Upload a CSV or Parquet file to check missing values, column types and distributions, then train a baseline model. Files are deleted after 30 minutes." />
    <meta property="og:title" content="Preflight: check a dataset before you model it" />
    <meta property="og:description" content="Check a CSV or Parquet file for gaps and column types, then train a baseline model." />
    <meta property="og:type" content="website" />
    <meta property="og:url" content="https://preflight.kooexperience.com/" />
    <link rel="icon" type="image/svg+xml" href="/assets/favicon.svg" />
    <link rel="preload" href="/assets/fonts/hanken-grotesk-latin-wght-normal.woff2" as="font" type="font/woff2" crossorigin />
    {%css%}
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
"""

# -----------------------
# Stores
# -----------------------
store_block = html.Div(
    [
        dcc.Store(id="store-df-json"),  # opaque token only; dataset is cached server-side
        dcc.Store(id="store-target"),
        dcc.Store(id="store-feature-types"),  # {col: type}
        dcc.Store(id="store-eda-page", data=0),  # EDA pagination state
        dcc.Store(id="store-upload-note"),
    ]
)

# -----------------------
# Panes (always mounted)
# -----------------------
guide_pane = html.Div(
    id="pane-guide",
    children=[
        dbc.Card(
            dbc.CardBody(
                [
                    html.H2("How to check a dataset"),
                    html.Ol(
                        [
                            html.Li("Tick the confirmation box, then upload a CSV or Parquet file."),
                            html.Li("Data health: check row and column counts, duplicates and missing values per column, then pick the column you want to predict."),
                            html.Li("Missing values: see which rows and columns have gaps, and whether they cluster."),
                            html.Li("Column types: confirm which columns are numbers, categories or IDs to drop. Save before you model."),
                            html.Li("Explore: chart each column against the target and look for strongly correlated pairs."),
                            html.Li("Model: train a baseline and read its scores on a held-out test split."),
                        ]
                    ),
                    html.P(
                        "A baseline is a quick yardstick, not a finished model. If a simple model already scores well, check the columns for leaks of the answer.",
                        className="pf-hint",
                    ),
                ]
            ),
            className="mt-3",
        )
    ],
)

health_pane = html.Div(
    id="pane-health",
    children=[
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.Div(id="health-rows"),
                            html.Div(id="health-cols"),
                            html.Div(id="health-dups"),
                            html.Div(id="health-miss-any"),
                        ],
                        className="pf-stats",
                    ),
                    html.H2("Column summary"),
                    dash_table.DataTable(
                        id="health-profile-table",
                        data=[],
                        columns=[
                            {"name": "Column", "id": "column"},
                            {"name": "Type", "id": "dtype"},
                            {"name": "Missing %", "id": "missing_%", "type": "numeric", "format": Format(precision=1, scheme=Scheme.fixed)},
                            {"name": "Unique values", "id": "n_unique", "type": "numeric", "format": Format(group=Group.yes)},
                            {"name": "Unique ratio", "id": "unique_ratio", "type": "numeric", "format": Format(precision=3, scheme=Scheme.fixed)},
                        ],
                        style_cell_conditional=[
                            {"if": {"column_id": c}, "textAlign": "left"} for c in ("column", "dtype")
                        ],
                        page_size=12,
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "fontFamily": "Hanken Grotesk, system-ui, sans-serif",
                            "fontSize": 12,
                            "padding": "6px",
                            "backgroundColor": "#ffffff",
                            "color": "#141a1d",
                            "border": "1px solid #dbe1e1",
                        },
                        style_header={
                            "fontWeight": "bold",
                            "backgroundColor": "#f0f3f3",
                            "border": "1px solid #dbe1e1",
                        },
                        css=[
                            {
                                "selector": ".dash-table-container .previous-next-container span",
                                "rule": "color: #141a1d !important;",
                            },
                            {
                                "selector": ".dash-table-container .previous-next-container .page-number",
                                "rule": "color: #141a1d !important;",
                            },
                            {
                                "selector": ".dash-table-container .previous-next-container input",
                                "rule": "color: #141a1d !important; background-color: #ffffff !important;",
                            },
                            {
                                "selector": ".dash-table-container .previous-next-container input[type='text']",
                                "rule": "-webkit-text-fill-color: #141a1d !important;",
                            },
                        ],
                    ),
                    html.Div(style={"height": "12px"}),
                    html.Label("Target column (what a model should predict)"),
                    dcc.Dropdown(id="target-dropdown", options=[], value=None, clearable=True),
                    html.Div(id="target-note", style={"marginTop": "8px"}),
                ]
            ),
            className="mt-3",
        )
    ],
)

missing_pane = html.Div(
    id="pane-missing",
    children=[
        dbc.Card(
            dbc.CardBody(
                [
                    html.H2("Missing values"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Sample rows"),
                                    dcc.Slider(
                                        id="missing-sample-n",
                                        min=50,
                                        max=50,
                                        step=50,
                                        value=50,
                                        updatemode="mouseup",
                                    ),
                                ],
                                md=8,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Seed"),
                                    dcc.Input(id="missing-seed", type="number", value=42, step=1, style={"width": "100%"}),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("Columns shown"),
                                    dcc.Dropdown(
                                        id="missing-col-mode",
                                        options=[
                                            {"label": "All columns", "value": "all"},
                                            {"label": "Top missing columns", "value": "top_missing"},
                                        ],
                                        value="all",
                                        clearable=False,
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("Top N (when applicable)"),
                                    dcc.Input(id="missing-top-n", type="number", value=35, min=5, step=1, style={"width": "100%"}),
                                ],
                                md=4,
                            ),
                        ],
                        className="mb-2",
                    ),
                    html.Div(id="missing-sample-label", className="mb-2"),
                    dcc.Graph(id="missing-matrix-fig", figure=placeholder_fig("Upload a dataset to see where values are missing.")),
                    dcc.Graph(id="missing-bar-fig", figure=placeholder_fig("Missing values per column appear here.")),
                ]
            ),
            className="mt-3",
        )
    ],
)

typing_pane = html.Div(
    id="pane-typing",
    children=[
        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Treat a column as a category when it has at most this many distinct values"),
                                    dcc.Slider(
                                        id="k-threshold",
                                        min=5,
                                        max=50,
                                        step=1,
                                        value=20,
                                        marks={v: str(v) for v in range(5, 51, 5)},
                                    ),
                                    html.Div(style={"height": "12px"}),
                                    html.Label("Drop a column as an ID when at least this share of its values is unique"),
                                    dcc.Slider(
                                        id="id-threshold",
                                        min=0.70,
                                        max=0.99,
                                        step=0.01,
                                        value=0.90,
                                        marks={round(v, 2): f"{v:.2f}" for v in np.arange(0.70, 1.00, 0.05)},
                                    ),
                                    html.Div(style={"height": "12px"}),
                                    dcc.Checklist(
                                        id="int-as-cat",
                                        options=[{"label": "Treat whole-number columns with few values as categories", "value": "on"}],
                                        value=["on"],
                                    ),
                                    html.Button("Recompute", id="btn-recompute", n_clicks=0),
                                    html.Div(id="typing-need-target", style={"marginTop": "12px"}),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                dcc.Loading(
                                    id="loading-typing",
                                    type="default",
                                    color="#0f766e",
                                    children=[
                                        html.H2("Column types"),
                                        dash_table.DataTable(
                                            id="typing-table",
                                            data=[],
                                            columns=[
                                                {"name": "Column", "id": "column", "editable": False},
                                                {"name": "Suggested type", "id": "suggested_type", "editable": False},
                                                {"name": "Your choice", "id": "override_type", "presentation": "dropdown", "editable": True},
                                            ],
                                            editable=True,
                                            dropdown={
                                                "override_type": {
                                                    "options": [{"label": t, "value": t} for t in ["numeric","ordinal", "categorical", "high_card_categorical", "drop"]]
                                                }
                                            },
                                            page_size=12,
                                            style_table={"overflowX": "auto", "maxHeight": "420px", "overflowY": "auto"},
                                            style_cell={
                                                "fontFamily": "Hanken Grotesk, system-ui, sans-serif",
                                                "fontSize": 12,
                                                "padding": "6px",
                                                "backgroundColor": "#ffffff",
                                                "color": "#141a1d",
                                                "border": "1px solid #dbe1e1",
                                                "overflow": "visible",
                                            },
                                            style_data={"overflow": "visible"},
                                            style_cell_conditional=[{"if": {"column_id": c}, "textAlign": "left"} for c in ("column", "suggested_type", "override_type")],
                                            style_header={"fontWeight": "bold", "backgroundColor": "#f0f3f3", "border": "1px solid #dbe1e1"},
                                            css=[
                                                {"selector": ".dash-spreadsheet-container .dropdown", "rule": "position: static !important;"},
                                                {"selector": ".dash-spreadsheet-container .Select-menu-outer", "rule": "display: block !important; z-index: 9999 !important;"},
                                                {"selector": ".dash-spreadsheet-container .Select-control", "rule": "background-color: #ffffff !important; color: #141a1d !important; border: 1px solid #8a9494 !important;"},
                                                {"selector": ".dash-spreadsheet-container .Select-value-label", "rule": "color: #141a1d !important;"},
                                                {"selector": ".dash-spreadsheet-container .Select-menu-outer *", "rule": "color: #141a1d !important;"},
                                            ],
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Button("Save types", id="btn-save-types", n_clicks=0), width="auto"),
                                                dbc.Col(html.Div(id="typing-save-note"), width=True),
                                            ],
                                            align="center",
                                            className="mt-2",
                                        ),
                                    ],
                                ),
                                md=8,
                            ),
                        ]
                    )
                ]
            ),
            className="mt-3",
        )
    ],
)

eda_pane = html.Div(
    id="pane-eda",
    children=[
        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Columns to chart"),
                                    dcc.Dropdown(id="eda-features", options=[], multi=True, value=[]),
                                    dbc.Row(
                                        [
                                            dbc.Col(html.Button("Select all", id="btn-eda-select-all", n_clicks=0), width="auto"),
                                            dbc.Col(html.Button("Clear", id="btn-eda-clear", n_clicks=0), width="auto"),
                                            dbc.Col(html.Div(id="eda-hint"), width=True),
                                        ],
                                        className="mt-2",
                                        align="center",
                                    ),
                                ],
                                md=8,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Categorical mode"),
                                    dcc.Dropdown(
                                        id="eda-cat-mode",
                                        options=[{"label": "Counts", "value": "counts"}, {"label": "Rate", "value": "rate"}],
                                        value="counts",
                                        clearable=False,
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    dcc.Checklist(
                                        id="eda-scatter-toggle",
                                        options=[{"label": "Show scatter when exactly 2 numeric features are selected", "value": "on"}],
                                        value=["on"],
                                    ),
                                ],
                                md=4,
                            ),
                        ],
                        className="mb-2",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(html.Label("Charts per page"), width="auto"),
                            dbc.Col(
                                dcc.Dropdown(
                                    id="eda-page-size",
                                    options=[{"label": str(v), "value": v} for v in [4, 6, 8, 10, 12]],
                                    value=8,
                                    clearable=False,
                                    style={"minWidth": "120px"},
                                ),
                                width="auto",
                            ),
                            dbc.Col(html.Button("Previous", id="btn-eda-prev", n_clicks=0), width="auto"),
                            dbc.Col(html.Button("Next", id="btn-eda-next", n_clicks=0), width="auto"),
                            dbc.Col(html.Div(id="eda-page-label"), width=True),
                        ],
                        className="mt-2",
                        align="center",
                    ),
                    dcc.Checklist(
                        id="eda-show-corr",
                        options=[{"label": "Show correlation section", "value": "on"}],
                        value=["on"],  # default ON; set to [] if you want default OFF
                        className="mb-2",
                    ),
                    dcc.Loading(
                        id="loading-eda",
                        type="default",
                        color="#0f766e",
                        children=[
                            html.Div(
                                id="corr-block",
                                children=[
                                    html.Hr(),
                                    html.H2("Correlation (numeric only)"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Label("Method"),
                                                    dcc.Dropdown(
                                                        id="corr-method",
                                                        options=[
                                                            {"label": "Spearman", "value": "spearman"},
                                                            {"label": "Pearson", "value": "pearson"},
                                                        ],
                                                        value="spearman",
                                                        clearable=False,
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label("Min |r|"),
                                                    dcc.Slider(
                                                        id="corr-min-r",
                                                        min=0.1,
                                                        max=0.95,
                                                        step=0.05,
                                                        value=0.5,
                                                    ),
                                                ],
                                                md=6,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label("Top pairs"),
                                                    dcc.Input(
                                                        id="corr-top-k",
                                                        type="number",
                                                        value=10,
                                                        min=1,
                                                        step=1,
                                                        style={"width": "100%"},
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                        ],
                                        className="mb-2",
                                    ),
                                    dcc.Graph(id="corr-heatmap", figure=placeholder_fig("Upload a dataset and pick a target to see correlations.")),
                                    html.Div(id="corr-scatter-container"),
                                ],
                            ),
                            html.Div(id="eda-multi-container"),
                        ],
                    ),
                ]
            ),
            className="mt-3",
        )
    ],
)

model_pane = html.Div(
    id="pane-model",
    children=[
        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Model"),
                                    dcc.Dropdown(
                                        id="model-choice",
                                        options=[
                                            {"label": "Dummy", "value": "dummy"},
                                            {"label": "Logistic Regression", "value": "logreg"},
                                            {"label": "Random Forest", "value": "rf"},
                                            {"label": "KNN", "value": "knn"},
                                            {"label": "SVM (RBF)", "value": "svm"},
                                        ],
                                        value="logreg",
                                        clearable=False,
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    html.Label("CV metric to optimize"),
                                    dcc.Dropdown(
                                        id="cv-metric",
                                        options=[{"label": k, "value": v} for k, v in (CV_METRICS + BINARY_ONLY_METRICS)],
                                        value="f1_macro",
                                        clearable=False,
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Train row cap (0 = full)"),
                                    dcc.Input(id="train-cap", type="number", value=0, min=0, step=500, style={"width": "100%"}),
                                ],
                                md=4,
                            ),
                        ],
                        className="mb-2",
                    ),
                    html.Label("Features to include"),
                    dcc.Dropdown(id="model-feature-include", options=[], value=[], multi=True),
                    dbc.Row(
                        [
                            dbc.Col(html.Button("Select all", id="btn-model-select-all", n_clicks=0), width="auto"),
                            dbc.Col(html.Button("Clear", id="btn-model-clear", n_clicks=0), width="auto"),
                            dbc.Col(html.Div(id="model-feature-hint"), width=True),
                        ],
                        className="mt-2",
                        align="center",
                    ),
                    html.Hr(),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Test size"),
                                    dcc.Slider(id="test-size", min=0.1, max=0.4, step=0.05, value=0.2),
                                    html.Div(style={"height": "6px"}),
                                    dcc.Checklist(
                                        id="scale-numeric",
                                        options=[{"label": "Scale numeric (recommended for LogReg/KNN/SVM)", "value": "on"}],
                                        value=["on"],
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Threshold (binary only)"),
                                    dcc.Slider(id="threshold", min=0.05, max=0.95, step=0.05, value=0.5),
                                    html.Div(style={"height": "6px"}),
                                    html.Button("Train", id="btn-train", n_clicks=0),
                                ],
                                md=6,
                            ),
                        ],
                        className="mb-2",
                    ),
                    dcc.Loading(
                        id="loading-train",
                        type="default",
                        color="#0f766e",
                        children=[
                            html.Div(id="model-metrics", className="mb-3"),
                            dbc.Row(
                                [
                                    dbc.Col(dcc.Graph(id="cm-graph", figure=placeholder_fig("Train a model to see the confusion matrix.")), md=6),
                                    dbc.Col(dcc.Graph(id="roc-graph", figure=placeholder_fig("Train a model to see the ROC curve.")), md=6),
                                ]
                            ),
                            html.Div(style={"height": "12px"}),
                            dcc.Graph(id="pr-graph", figure=placeholder_fig("Train a model to see the precision-recall curve.")),
                        ],
                    ),
                ]
            ),
            className="mt-3",
        )
    ],
)

# -----------------------
# Layout
# -----------------------
app.layout = dbc.Container(
    fluid=True,
    children=[
        html.Header(
            [
                html.A(
                    [
                        html.Img(src="/assets/favicon.svg", alt="", width=24, height=24),
                        html.Span("Preflight"),
                    ],
                    href="/",
                    className="pf-logo",
                ),
                html.Nav(
                    [html.A("GitHub", href="https://github.com/haomingkoo/preflight", rel="noopener")],
                    className="pf-nav-links",
                    **{"aria-label": "Main"},
                ),
            ],
            className="pf-nav",
        ),
        html.Section(
            [
                html.H1("Check a dataset before you model it", className="pf-hero-title"),
                html.P(
                    "Upload a CSV or Parquet file to see its missing values, column types and distributions, then train a baseline model.",
                    className="pf-hero-sub",
                ),
                html.Button("▶ 30-second tour", id="tour-open", type="button"),
            ],
            className="pf-hero",
        ),
        html.Div(
            [
                html.P("Upload a file", className="pf-upload-title"),
                dcc.Checklist(
                    id="upload-consent",
                    options=[{"label": CONSENT_TEXT, "value": CONSENT_VALUE}],
                    value=[],
                    className="pf-consent",
                ),
                dcc.Upload(
                    id="upload-data",
                    children=html.Div([
                        html.Span(["Drag a file here or ", html.A("browse")], className="pf-upload-main"),
                        html.Span(f"CSV or Parquet, up to {MAX_UPLOAD_MB} MB", className="pf-upload-meta"),
                        html.Span("Tick the box above to turn on upload.", className="pf-upload-locked"),
                    ]),
                    accept=".csv,.parquet",
                    max_size=MAX_UPLOAD_BYTES,
                    multiple=False,
                    disabled=True,
                    # dcc.Upload fades to 0.55 opacity when disabled; keep the locked box readable.
                    style_disabled={"opacity": 1},
                ),
                html.Div(
                    [
                        html.P(
                            [
                                "Your file is processed on our server (Railway, behind Cloudflare) only to produce this analysis. "
                                f"It is deleted {DATA_RETENTION_SECONDS // 60} minutes after upload, or right away if you use "
                                "Delete my data now. It is never shared or used for anything else. ",
                                html.A("Privacy", href="/privacy", target="_blank", rel="noopener"),
                                " · ",
                                html.A("Terms", href="/terms", target="_blank", rel="noopener"),
                            ],
                            className="pf-upload-notice",
                        ),
                        html.Button("Delete my data now", id="btn-delete-data", n_clicks=0, className="pf-delete-btn"),
                    ],
                    className="pf-upload-row",
                ),
            ],
            className="pf-upload-block",
        ),
        dcc.Loading(
            id="loading-upload",
            type="default",
            color="#0f766e",
            children=html.Div(
                [
                    html.Div(id="upload-loading-anchor", style={"display": "none"}),
                    html.Div(id="upload-note", className="mt-2"),
                ]
            ),
        ),
        html.Details(
            [
                html.Summary("Advanced: missing value tokens"),
                html.Div(
                    [
                        html.Label("Missing tokens (comma-separated)", htmlFor="missing-tokens"),
                        dcc.Input(
                            id="missing-tokens",
                            type="text",
                            value=",".join(MISSING_TOKENS_DEFAULT),
                            style={"width": "100%"},
                        ),
                        html.P("Cells matching these values are treated as missing. Set them before you upload.", className="pf-hint"),
                    ],
                    className="pf-advanced-body",
                ),
            ],
            className="pf-advanced",
        ),
        store_block,
        dcc.Tabs(
            id="tabs",
            value="tab-guide",
            children=[
                dcc.Tab(label="1. Guide", value="tab-guide"),
                dcc.Tab(label="2. Data health", value="tab-health"),
                dcc.Tab(label="3. Missing values", value="tab-missing"),
                dcc.Tab(label="4. Column types", value="tab-typing"),
                dcc.Tab(label="5. Explore", value="tab-eda"),
                dcc.Tab(label="6. Model", value="tab-model"),
            ],
        ),
        html.Div([guide_pane, health_pane, missing_pane, typing_pane, eda_pane, model_pane], style={"paddingTop": "12px"}),
        html.Footer(
            [
                html.Span("© 2026 Haoming Koo"),
                html.Span("·", className="pf-sep", **{"aria-hidden": "true"}),
                html.A("Privacy", href="/privacy"),
                html.Span("·", className="pf-sep", **{"aria-hidden": "true"}),
                html.A("Terms", href="/terms"),
                html.Span("·", className="pf-sep", **{"aria-hidden": "true"}),
                html.Span(["Contact (", html.A("privacy@kooexperience.com", href="mailto:privacy@kooexperience.com"), ")"]),
                html.Span("·", className="pf-sep", **{"aria-hidden": "true"}),
                html.Span(["More projects → ", html.A("kooexperience.com", href="https://kooexperience.com/")]),
            ],
            className="pf-footer",
        ),
        # Behaviour (open, close, ?tour=1) lives in assets/tour.js.
        html.Dialog(
            [
                html.Div(
                    [
                        html.H2("Preflight in 30 seconds", id="tour-title", className="pf-tour-title"),
                        html.Button("×", id="tour-close", type="button", **{"aria-label": "Close video"}),
                    ],
                    className="pf-tour-head",
                ),
                html.Video(
                    [
                        html.Source(src="/assets/media/preflight.webm", type="video/webm"),
                        html.Source(src="/assets/media/preflight.mp4", type="video/mp4"),
                    ],
                    id="tour-video",
                    controls=True,
                    preload="none",
                    poster="/assets/media/preflight-poster.jpg",
                    className="pf-tour-video",
                ),
                html.P("Music: Sascha Ende, ende.app (CC BY 4.0)", className="pf-tour-credit"),
            ],
            id="tour-dialog",
            className="pf-tour",
            **{"aria-labelledby": "tour-title"},
        ),
    ],
)

# -----------------------
# Show/hide panes
# -----------------------
@app.callback(
    Output("pane-guide", "style"),
    Output("pane-health", "style"),
    Output("pane-missing", "style"),
    Output("pane-typing", "style"),
    Output("pane-eda", "style"),
    Output("pane-model", "style"),
    Input("tabs", "value"),
)
def show_hide(tab):
    show = {"display": "block"}
    hide = {"display": "none"}
    return (
        show if tab == "tab-guide" else hide,
        show if tab == "tab-health" else hide,
        show if tab == "tab-missing" else hide,
        show if tab == "tab-typing" else hide,
        show if tab == "tab-eda" else hide,
        show if tab == "tab-model" else hide,
    )

# -----------------------
# EDA: show/hide correlation block
# -----------------------
@app.callback(
    Output("corr-block", "style"),
    Input("eda-show-corr", "value"),
)
def toggle_corr_block(show_vals):
    if "on" in (show_vals or []):
        return {"display": "block"}
    return {"display": "none"}

# -----------------------
# Upload: store dataset token + reset target
# (DO NOT write store-feature-types here. One writer only.)
# -----------------------
@app.callback(
    Output("store-df-json", "data"),
    Output("target-dropdown", "value"),  # reset UI, NOT the store
    Output("store-upload-note", "data"),
    Output("upload-loading-anchor", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("missing-tokens", "value"),
    State("upload-consent", "value"),
)
def on_upload(contents, filename, missing_tokens_str, consent=None):
    if not contents or not filename:
        # Keep any message already shown (e.g. the delete confirmation).
        return None, None, no_update, ""

    # Enforced here as well as in the UI: a crafted callback request without consent is refused.
    if CONSENT_VALUE not in (consent or []):
        return None, None, ("error", CONSENT_REQUIRED_MESSAGE), ""

    if not _allow_action("upload", UPLOAD_RATE_LIMIT_PER_MIN):
        return None, None, ("error", "Too many uploads in a short time. Wait a minute, then try again."), ""

    try:
        df = parse_contents(contents, filename)
    except ValueError as e:
        return None, None, ("error", str(e)), ""
    except Exception as exc:
        log_error("Unexpected upload parsing error", exc)
        return None, None, ("error", "We could not read that file. Check it is a CSV or Parquet file and try again."), ""

    tokens = [t.strip() for t in (missing_tokens_str or "").split(",")]
    df = normalize_missing_tokens(df, tokens)

    dup_count = len(df.columns) - len(set(df.columns))
    df = make_unique_columns(df)
    df = enforce_dataframe_limits(df)

    notes = []
    if dup_count > 0:
        notes.append(f"{dup_count} duplicate column name(s) auto-renamed.")
    if len(df) > MAX_ROWS_WARN:
        notes.append(
            f"{len(df):,} rows loaded — large datasets may be slow. "
            f"Consider a sampled subset for EDA."
        )

    rows, cols = df.shape
    summary = f"Loaded {rows:,} rows and {cols:,} columns from {filename}."
    if notes:
        note = ("warning", f"{summary} {' '.join(notes)}")
    else:
        note = ("success", summary)

    try:
        dataset_token = save_df_for_session(df)
    except ValueError as e:
        return None, None, ("error", str(e)), ""
    except Exception as exc:
        log_error("Failed to persist uploaded dataset", exc)
        return None, None, ("error", "The upload did not finish. Try again."), ""

    # Explicit memory cleanup after dataframe is saved to disk
    del df
    gc.collect()
    
    return dataset_token, None, note, filename

@app.callback(
    Output("upload-data", "disabled"),
    Input("upload-consent", "value"),
)
def toggle_upload_enabled(consent):
    return CONSENT_VALUE not in (consent or [])


@app.callback(
    Output("store-df-json", "data", allow_duplicate=True),
    Output("store-upload-note", "data", allow_duplicate=True),
    Output("upload-data", "contents"),
    Input("btn-delete-data", "n_clicks"),
    prevent_initial_call=True,
)
def on_delete_data(n_clicks):
    if not n_clicks:
        return no_update, no_update, no_update
    removed = delete_session_dataset()
    gc.collect()
    msg = "Your data was deleted from the server." if removed else "There was no data stored for this session."
    # Clearing contents also lets the same file be uploaded again.
    return None, ("success", msg), None


# The browser ignores re-selecting the same file unless the input is cleared.
app.clientside_callback(
    """
    function(n) {
        const input = document.querySelector('#upload-data input[type=file]');
        if (input) { input.value = ''; }
        return null;
    }
    """,
    Output("upload-data", "filename"),
    Input("btn-delete-data", "n_clicks"),
    prevent_initial_call=True,
)


@app.callback(
    Output("upload-note", "children"),
    Input("store-upload-note", "data"),
)
def show_upload_note(note):
    if not note:
        return ""
    level, msg = note
    if level == "error":
        color = "danger"
    elif level == "success":
        color = "success"
    else:
        color = "warning"
    return dbc.Alert(msg, color=color, dismissable=True)

# -----------------------
# Data Health: profile + dropdown options
# -----------------------
def _stat(label: str, value: str) -> list:
    return [html.Span(label, className="pf-stat-label"), html.Span(value, className="pf-stat-value")]


@app.callback(
    Output("health-rows", "children"),
    Output("health-cols", "children"),
    Output("health-dups", "children"),
    Output("health-miss-any", "children"),
    Output("health-profile-table", "data"),
    Output("target-dropdown", "options"),
    Input("store-df-json", "data"),
)
def update_health(df_json):
    df = df_from_store(df_json)
    if df is None:
        empty = html.P("Upload a file to see its rows, columns, duplicates and missing values.", className="pf-empty")
        return empty, "", "", "", [], []

    prof = basic_profile(df)
    n_rows, n_cols = df.shape
    dup = int(df.duplicated().sum())
    miss_any = float(df.isna().any(axis=1).mean() * 100)

    result = (
        _stat("Rows", f"{n_rows:,}"),
        _stat("Columns", f"{n_cols:,}"),
        _stat("Duplicate rows", f"{dup:,}"),
        _stat("Rows with any missing", f"{miss_any:.1f}%"),
        prof.to_dict("records"),
        [{"label": c, "value": c} for c in df.columns],
    )
    
    return result

# -----------------------
# Target selection (DO NOT reset store-feature-types here)
# -----------------------
@app.callback(
    Output("store-target", "data"),
    Output("target-note", "children"),
    Input("target-dropdown", "value"),
    State("store-df-json", "data"),
)
def set_target(target, df_json):
    df = df_from_store(df_json)
    if df is None or not target:
        return None, ""

    nunique = df[target].nunique(dropna=True)
    note = f"Target selected: {target}. Unique classes (non-null): {nunique}."
    if nunique > 20:
        note += " This looks high-cardinality. Confirm target selection."

    return target, note

# -----------------------
# Feature types controller (THE ONLY WRITER of store-feature-types + typing-save-note)
# Resets on upload/target changes, saves on Save Types.
# -----------------------
@app.callback(
    Output("store-feature-types", "data"),
    Output("typing-save-note", "children"),
    Input("store-df-json", "data"),
    Input("store-target", "data"),
    Input("btn-save-types", "n_clicks"),
    State("typing-table", "data"),
)
def feature_types_controller(df_json, target, n_save, typing_rows):
    triggered = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""

    # dataset/target changed => invalidate feature types
    if triggered in ("store-df-json.data", "store-target.data"):
        return None, ""

    # Save Types clicked
    if triggered == "btn-save-types.n_clicks":
        if not df_json:
            return None, "Upload a file first."
        if not target:
            return None, "Pick a target column in Data health first."
        if not typing_rows:
            return None, "The column types table is empty."
        
        types = {
            r["column"]: (r.get("override_type") or r.get("suggested_type") or "drop")
            for r in typing_rows
        }
        # types = {r["column"]: r.get("override_type", r.get("suggested_type", "drop")) for r in typing_rows}
        return types, f"Saved types for {len(types)} columns."

    return no_update, no_update

# -----------------------
# Missingness: slider ranges
# -----------------------
@app.callback(
    Output("missing-sample-n", "max"),
    Output("missing-sample-n", "value"),
    Output("missing-sample-n", "step"),
    Output("missing-sample-n", "marks"),
    Input("store-df-json", "data"),
)
def configure_missing_slider(df_json):
    df = df_from_store(df_json)
    if df is None or len(df) == 0:
        return 50, 50, 50, {50: "50"}

    n = len(df)
    step = max(50, n // 200) if n > 0 else 50
    default_val = min(5000, n)
    marks = {50: "50", default_val: str(default_val), n: f"{n:,}"} if n >= 50 else {n: f"{n:,}"}
    return max(50, n), default_val, step, marks

@app.callback(
    Output("missing-matrix-fig", "figure"),
    Output("missing-bar-fig", "figure"),
    Output("missing-sample-label", "children"),
    Input("store-df-json", "data"),
    Input("missing-sample-n", "value"),
    Input("missing-seed", "value"),
    Input("missing-col-mode", "value"),
    Input("missing-top-n", "value"),
)
def update_missingness(df_json, sample_n, seed, col_mode, top_n):
    df = df_from_store(df_json)
    if df is None:
        return (
            placeholder_fig("Upload a dataset to see where values are missing."),
            placeholder_fig("Missing values per column appear here."),
            "",
        )

    seed = int(seed or 42)
    sample_n = int(sample_n or min(5000, len(df)))
    top_n = int(top_n or 35)

    matrix_fig = missingness_heatmap_figure(df, sample_n, seed, col_mode or "all", top_n)

    miss_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    miss_pct = miss_pct[miss_pct > 0].head(25)

    if len(miss_pct) == 0:
        bar_fig = go.Figure()
        bar_fig.update_layout(height=CHART_HEIGHT, template="preflight", title="Top Missing Columns (up to 25)")
    else:
        bar_fig = px.bar(
            x=miss_pct.values,
            y=miss_pct.index,
            orientation="h",
            labels={"x": "Missing %", "y": "Column"},
            title="Top Missing Columns (up to 25)",
            template="preflight",
        )
        bar_fig.update_yaxes(categoryorder="total ascending")
        bar_fig.update_layout(height=CHART_HEIGHT, margin=dict(l=220, r=40, t=60, b=40))

    label = f"Matrix uses {min(sample_n, len(df)):,} sampled rows (seed={seed}). Hover shows full column names."
    return matrix_fig, bar_fig, label

# -----------------------
# Typing: populate typing table
# -----------------------
@app.callback(
    Output("typing-table", "data"),
    Output("typing-need-target", "children"),
    Input("store-df-json", "data"),
    Input("store-target", "data"),
    Input("btn-recompute", "n_clicks"),
    State("k-threshold", "value"),
    State("id-threshold", "value"),
    State("int-as-cat", "value"),
)
def typing_table_controller(df_json, target, _n_clicks, k, id_thr, int_as_cat):
    df = df_from_store(df_json)
    if df is None:
        return [], "Upload a file first."
    if not target:
        return [], "Pick a target column in Data health first."

    cfg = TypingConfig(
        low_card_threshold=int(k or 20),
        id_unique_ratio_threshold=float(id_thr or 0.90),
        treat_small_unique_int_as_cat=("on" in (int_as_cat or [])),
    )
    suggested = auto_type_columns(df, target, cfg)
    rows = [{"column": c, "suggested_type": t, "override_type": t} for c, t in suggested.items()]
    return rows, ""

# -----------------------
# EDA: options + select all/clear + charts (incl scatter)
# -----------------------
@app.callback(
    Output("eda-features", "options"),
    Input("store-df-json", "data"),
    Input("store-target", "data"),
)
def update_eda_options(df_json, target):
    df = df_from_store(df_json)
    if df is None:
        return []
    cols = [c for c in df.columns if c != target] if target else list(df.columns)
    return [{"label": c, "value": c} for c in cols]

@app.callback(
    Output("eda-features", "value"),
    Output("eda-hint", "children"),
    Input("btn-eda-select-all", "n_clicks"),
    Input("btn-eda-clear", "n_clicks"),
    State("eda-features", "options"),
    State("eda-features", "value"),
    prevent_initial_call=True,
)
def eda_select_all_clear(_n_all, _n_clear, options, current):
    triggered = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
    all_vals = [o["value"] for o in (options or [])]

    if triggered.startswith("btn-eda-select-all"):
        return all_vals, f"{len(all_vals)} selected."
    if triggered.startswith("btn-eda-clear"):
        return [], "0 selected."

    return current or [], ""

@app.callback(
    Output("store-eda-page", "data"),
    Output("eda-page-label", "children"),
    Input("btn-eda-prev", "n_clicks"),
    Input("btn-eda-next", "n_clicks"),
    Input("eda-features", "value"),
    Input("eda-page-size", "value"),
    State("store-eda-page", "data"),
)
def eda_page_controller(n_prev, n_next, features, page_size, page):
    triggered = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
    features = features or []
    page_size = int(page_size or 8)
    page = int(page or 0)

    total = len(features)
    max_page = max(0, (total - 1) // page_size) if total > 0 else 0

    if triggered == "eda-features.value":
        page = 0
    elif triggered.startswith("btn-eda-prev"):
        page = max(0, page - 1)
    elif triggered.startswith("btn-eda-next"):
        page = min(max_page, page + 1)

    start = page * page_size + 1 if total else 0
    end = min((page + 1) * page_size, total)

    label = (
        f"Showing {start} to {end} of {total} columns, page {page + 1} of {max_page + 1}"
        if total else "No columns picked yet."
    )
    return page, label

@app.callback(
    Output("eda-multi-container", "children"),
    Input("eda-features", "value"),
    Input("eda-cat-mode", "value"),
    Input("eda-scatter-toggle", "value"),
    Input("store-eda-page", "data"),
    Input("eda-page-size", "value"),
    Input("store-feature-types", "data"),
    State("store-df-json", "data"),
    State("store-target", "data"),
)
def update_eda_multi(features, cat_mode, scatter_toggle, page, page_size, feature_types, df_json, target,):
    df = df_from_store(df_json)
    if df is None:
        return html.Div("Upload a file first.")
    if not target:
        return html.Div("Pick a target column in Data health first.")
    
    if feature_types:
        df = apply_type_overrides(df, feature_types) 

    features = features or []
    features = [
        f for f in features
        if feature_types is None or feature_types.get(f) != "drop"
    ]
    if len(features) == 0:
        return html.Div("Select one or more features.")

    blocks = []

    if "on" in (scatter_toggle or []):
        numeric_selected = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
        if len(numeric_selected) == 2:
            x, y = numeric_selected
            fig_sc = px.scatter(df, x=x, y=y, color=target, template="preflight")
            fig_sc.update_layout(height=CHART_HEIGHT, title=f"Scatter: {x} vs {y} by {target}", margin=dict(l=40, r=40, t=60, b=40))
            blocks.append(dbc.Row([dbc.Col(dcc.Graph(figure=fig_sc), md=12)], className="mb-2"))

    page = int(page or 0)
    page_size = int(page_size or 8)

    start = page * page_size
    end = start + page_size
    page_features = features[start:end]

    per_feat_cols = []
    for feat in page_features:
        if feat not in df.columns or feat == target:
            continue

        if pd.api.types.is_numeric_dtype(df[feat]):
            fig = px.histogram(df, x=feat, color=target, barmode="overlay", marginal="box", template="preflight")
            fig.update_layout(height=CHART_HEIGHT, title=f"{feat} distribution by {target}", margin=dict(l=40, r=40, t=60, b=40))
            per_feat_cols.append(dbc.Col(dcc.Graph(figure=fig), md=6))
        else:
            tmp = df[[feat, target]].dropna()
            if tmp.empty:
                fig = go.Figure()
                fig.update_layout(height=CHART_HEIGHT, template="preflight", title=f"{feat}: no non-null rows")
                per_feat_cols.append(dbc.Col(dcc.Graph(figure=fig), md=6))
                continue

            if cat_mode == "counts":
                g = tmp.groupby([feat, target]).size().reset_index(name="count")
                g["feat_short"] = g[feat].astype(str).map(lambda x: shorten_label(x, 26))
                fig = px.bar(
                    g,
                    x="feat_short",
                    y="count",
                    color=target,
                    barmode="group",
                    template="preflight",
                    title=f"{feat} counts by {target}",
                    hover_data={feat: True, "feat_short": False, "count": True, target: True},
                )
            else:
                g = (
                    tmp.groupby(feat)[target]
                    .value_counts(normalize=True)
                    .rename("rate")
                    .reset_index()
                )
                g["feat_short"] = g[feat].astype(str).map(lambda x: shorten_label(x, 26))
                fig = px.bar(
                    g,
                    x="feat_short",
                    y="rate",
                    color=target,
                    barmode="stack",
                    template="preflight",
                    title=f"{feat} rate by category",
                    hover_data={feat: True, "feat_short": False, "rate": True, target: True},
                )

            fig.update_layout(height=CHART_HEIGHT, margin=dict(l=40, r=40, t=60, b=140))
            fig.update_xaxes(tickangle=45)
            per_feat_cols.append(dbc.Col(dcc.Graph(figure=fig), md=6))

    rows = [dbc.Row(per_feat_cols[i:i + 2], className="mb-2") for i in range(0, len(per_feat_cols), 2)]
    blocks.extend(rows)
    return html.Div(blocks)

@app.callback(
    Output("corr-heatmap", "figure"),
    Output("corr-scatter-container", "children"),
    Input("store-df-json", "data"),
    Input("store-target", "data"),
    Input("eda-show-corr", "value"),
    Input("corr-method", "value"),
    Input("corr-min-r", "value"),
    Input("corr-top-k", "value"),
    Input("store-feature-types", "data"),
)

def update_corr(df_json, target, show_vals, method, min_r, top_k, feature_types):
    if "on" not in (show_vals or []):
        return placeholder_fig("Tick Show correlation section to see it."), []

    df = df_from_store(df_json)
    if df is None:
        return placeholder_fig("Upload a file to see which numeric columns move together."), []
    if not target:
        return placeholder_fig("Pick a target column in Data health first."), []

    feature_types = feature_types or {}

    # Apply overrides (keeps your EDA behavior consistent)
    df = apply_type_overrides(df, feature_types)

    # Type-driven inclusion: only numeric (and ordinal, if you want)
    include_types = {"numeric", "ordinal"}  # remove "ordinal" if you want it excluded
    corr_cols = [c for c, t in feature_types.items() if t in include_types and c in df.columns]

    # Safety: in case feature_types misses some cols, do not auto-include them
    # If you want fallback behavior, tell me, but this keeps it strict.

    if len(corr_cols) < 2:
        return placeholder_fig("Correlation needs at least two numeric or ordinal columns. Check Column types."), []

    # Ensure columns are numeric for correlation computation
    num_df = df[corr_cols].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    if num_df.shape[1] < 2:
        return placeholder_fig("Correlation needs at least two numeric columns with values."), []

    method = method or "spearman"
    corr = num_df.corr(method=method)

    fig_hm = px.imshow(
        corr,
        aspect="auto",
        template="preflight",
        title=f"{method.title()} correlation (typed numeric only)",
        color_continuous_scale="BrBG",
        zmin=-1,
        zmax=1,
    )
    fig_hm.update_layout(height=CHART_HEIGHT, margin=dict(l=40, r=40, t=60, b=40))

    min_r = float(min_r or 0.5)
    top_k = int(top_k or 10)

    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) >= min_r:
                pairs.append((cols[i], cols[j], float(r)))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    pairs = pairs[:top_k]

    scatters = []
    for a, b, r in pairs:
        tmp = df[[a, b, target]].dropna()
        if tmp.empty:
            continue
        fig_sc = px.scatter(tmp, x=a, y=b, color=target, template="preflight")
        fig_sc.update_layout(height=CHART_HEIGHT, title=f"{a} vs {b} | r={r:.3f}", margin=dict(l=40, r=40, t=60, b=40))
        scatters.append(dbc.Row([dbc.Col(dcc.Graph(figure=fig_sc), md=12)], className="mb-2"))

    return fig_hm, scatters

# -----------------------
# Model: populate + select-all/clear
# -----------------------
@app.callback(
    Output("model-feature-include", "options"),
    Output("model-feature-include", "value"),
    Output("model-feature-hint", "children"),
    Input("store-df-json", "data"),
    Input("store-target", "data"),
    Input("store-feature-types", "data"),
    Input("btn-model-select-all", "n_clicks"),
    Input("btn-model-clear", "n_clicks"),
    State("model-feature-include", "value"),
)
def model_feature_controller(df_json, target, feature_types, _n_all, _n_clear, current_value):
    df = df_from_store(df_json)
    if df is None:
        return [], [], "Upload a file first."
    if not target:
        return [], [], "Pick a target column in Data health first."
    if not feature_types:
        return [], [], "Open Column types and press Save types before you train."

    X_cols = [c for c in df.columns if c != target]
    keep_cols = [c for c in X_cols if feature_types.get(c, "drop") != "drop"]

    opts = [{"label": c, "value": c} for c in keep_cols]
    all_values = [o["value"] for o in opts]

    triggered = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""

    if triggered.startswith("btn-model-select-all"):
        return opts, all_values, f"{len(all_values)} features selected (all)."

    if triggered.startswith("btn-model-clear"):
        return opts, [], "0 features selected."

    if current_value:
        valid = set(all_values)
        filtered = [v for v in current_value if v in valid]
        if filtered:
            return opts, filtered, f"{len(filtered)} features selected."

    return opts, all_values, f"{len(all_values)} features selected (default all)."

# -----------------------
# Train model
# -----------------------
@app.callback(
    Output("model-metrics", "children"),
    Output("cm-graph", "figure"),
    Output("roc-graph", "figure"),
    Output("pr-graph", "figure"),
    Input("btn-train", "n_clicks"),
    State("store-df-json", "data"),
    State("store-target", "data"),
    State("store-feature-types", "data"),
    State("model-choice", "value"),
    State("cv-metric", "value"),
    State("model-feature-include", "value"),
    State("test-size", "value"),
    State("scale-numeric", "value"),
    State("threshold", "value"),
    State("train-cap", "value"),
    prevent_initial_call=True,
    running=[
        (Output("btn-train", "disabled"), True, False),
        (Output("btn-train", "children"), "Training…", "Train"),
    ],
)
def train_model(
    _,
    df_json,
    target,
    feature_types,
    model_choice,
    cv_metric,
    include_features,
    test_size,
    scale_numeric_val,
    threshold,
    train_cap,
):
    empty_fig = placeholder_fig("Train a model to see this chart.")

    if not _allow_action("train", TRAIN_RATE_LIMIT_PER_MIN):
        return (
            html.Div("Too many training runs in a short time. Wait a minute, then try again."),
            empty_fig,
            empty_fig,
            empty_fig,
        )

    df = df_from_store(df_json)
    if df is None:
        return html.Div("Upload a file first."), empty_fig, empty_fig, empty_fig
    if not target:
        return html.Div("Pick a target column in Data health first."), empty_fig, empty_fig, empty_fig
    if not feature_types:
        return html.Div("Open Column types and press Save types first."), empty_fig, empty_fig, empty_fig
    if target not in df.columns:
        return html.Div("The target column is not in this file. Pick another in Data health."), empty_fig, empty_fig, empty_fig

    y_raw = df[target]
    X = df.drop(columns=[target])

    keep_cols = [c for c in X.columns if feature_types.get(c, "drop") != "drop"]
    include_features = include_features or keep_cols
    keep_cols = [c for c in keep_cols if c in set(include_features)]
    if len(keep_cols) == 0:
        return html.Div("No features selected. Use Select all or pick some features."), empty_fig, empty_fig, empty_fig
    if len(keep_cols) > MAX_TRAIN_FEATURES:
        return (
            html.Div(
                f"Too many features selected ({len(keep_cols)}). Maximum allowed is {MAX_TRAIN_FEATURES}."
            ),
            empty_fig,
            empty_fig,
            empty_fig,
        )
    X = X[keep_cols]

    mask = y_raw.notna()
    X = X.loc[mask]
    y_raw = y_raw.loc[mask]

    requested_cap = int(train_cap or 0)
    requested_cap = max(requested_cap, 0)
    hard_cap = MAX_TRAIN_ROWS_SVM if model_choice == "svm" else MAX_TRAIN_ROWS
    effective_cap = hard_cap
    if requested_cap > 0:
        effective_cap = min(requested_cap, hard_cap)

    cap_note = ""
    if requested_cap > hard_cap:
        cap_note = f"Requested train cap {requested_cap:,} was reduced to server limit {hard_cap:,}."

    if len(X) > effective_cap:
        X = X.sample(effective_cap, random_state=RANDOM_STATE)
        y_raw = y_raw.loc[X.index]
        if cap_note:
            cap_note += " "
        cap_note += f"Rows were sampled to {effective_cap:,} for training safety."

    classes = sorted(list(pd.Series(y_raw).dropna().unique()))
    if len(classes) < 2:
        return html.Div("The target needs at least two different values to train a model."), empty_fig, empty_fig, empty_fig
    is_binary = len(classes) == 2

    valid_metrics = {m for _, m in (CV_METRICS + BINARY_ONLY_METRICS)}
    if cv_metric not in valid_metrics:
        cv_metric = "f1_macro"
    if (cv_metric in {"roc_auc", "average_precision"}) and not is_binary:
        cv_metric = "f1_macro"

    if is_binary:
        class_to_int = {classes[0]: 0, classes[1]: 1}
        y = y_raw.map(class_to_int).astype(int).values
        label_names = [str(classes[0]), str(classes[1])]
    else:
        y = pd.factorize(y_raw)[0]
        label_names = [str(c) for c in classes]

    test_size = float(test_size or 0.2)
    test_size = min(max(test_size, 0.1), 0.4)
    threshold = float(threshold or 0.5)
    threshold = min(max(threshold, 0.05), 0.95)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )
    except ValueError:
        return (
            html.Div("Could not split data for training. Add more rows per class or adjust the dataset."),
            empty_fig,
            empty_fig,
            empty_fig,
        )

    scale_numeric = "on" in (scale_numeric_val or [])
    if model_choice in {"knn", "svm"}:
        scale_numeric = True

    feature_types_sel = {c: feature_types.get(c, "drop") for c in keep_cols}

    preprocessor, num_cols, ord_cols, cat_cols = build_preprocessor(
        feature_types_sel,
        scale_numeric=scale_numeric,
        available_cols=keep_cols,
    )
    
    if model_choice == "dummy":
        clf = DummyClassifier(strategy="most_frequent")
    elif model_choice == "logreg":
        clf = LogisticRegression(max_iter=4000, class_weight="balanced")
    elif model_choice == "rf":
        clf = RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        )
    elif model_choice == "knn":
        clf = KNeighborsClassifier(n_neighbors=15)
    else:
        clf = SVC(kernel="rbf", probability=True, class_weight="balanced")

    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
    class_counts = pd.Series(y_train).value_counts()
    if class_counts.empty or int(class_counts.min()) < 2:
        return (
            html.Div("Not enough rows per class for stratified CV after preprocessing."),
            empty_fig,
            empty_fig,
            empty_fig,
        )

    n_splits = min(5, int(class_counts.min()))
    n_splits = max(2, n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    try:
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=cv_metric)
        pipe.fit(X_train, y_train)
    except Exception as exc:
        log_error("Model training failed", exc)
        return (
            html.Div("Training failed. Try fewer rows/features or a simpler model."),
            empty_fig,
            empty_fig,
            empty_fig,
        )

    y_proba = None
    if hasattr(pipe, "predict_proba") and is_binary:
        y_proba = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = pipe.predict(X_test)

    rep = classification_report(y_test, y_pred, target_names=label_names, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    cm_fig = fig_confusion_matrix(cm, labels=label_names)

    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)

        if is_binary:
            y_proba = proba[:, 1]
            roc_fig, pr_fig, roc_auc, ap = fig_roc_pr(y_test, y_proba)
            roc_note = f"ROC-AUC={roc_auc:.3f}, Average Precision={ap:.3f}"

        else:
            # Multiclass ROC (One-vs-Rest), macro average
            y_bin = label_binarize(y_test, classes=np.arange(len(label_names)))
            roc_auc_macro = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")

            roc_fig = go.Figure()
            for i, name in enumerate(label_names):
                fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=str(name)))

            roc_fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash"))
            )

            roc_fig.update_layout(
                height=CHART_HEIGHT,
                title=f"ROC Curve (OvR, macro AUC={roc_auc_macro:.3f})",
                template="preflight",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                margin=dict(l=40, r=40, t=60, b=40),
            )

            # Keep PR empty for now (optional to implement later)
            pr_fig = placeholder_fig("The precision-recall curve is shown for binary targets only.")

            roc_note = f"Multiclass ROC-AUC (OvR macro)={roc_auc_macro:.3f}"
    else:
        no_proba = "This model does not output probabilities, so this curve is unavailable."
        roc_fig = placeholder_fig(no_proba)
        pr_fig = placeholder_fig(no_proba)
        roc_note = "ROC and PR curves need a model that outputs probabilities."

    metrics_block = html.Div(
        [
            html.H3("Metrics"),
            html.Div(
                [
                    html.Div(f"Rows used: {len(X):,}"),
                    html.Div(
                        f"Features used: {len(keep_cols)} | Numeric: {len(num_cols)} | "
                        f"Ordinal: {len(ord_cols)} | Categorical: {len(cat_cols)}"
                    ),
                    html.Div(f"CV folds: {n_splits}"),
                    html.Div(f"CV metric: {cv_metric} | mean={cv_scores.mean():.3f}, std={cv_scores.std():.3f}"),
                    html.Div(cap_note) if cap_note else html.Div(),
                    html.Div(roc_note),
                ],
                className="pf-metrics-list",
            ),
            html.H3("Classification report (test split)"),
            html.Pre(rep, style={"whiteSpace": "pre-wrap"}),
        ],
        className="pf-metrics",
    )

    # Explicit memory cleanup - most important callback for memory usage
    del df, X, y_raw, X_train, X_test, y_train, y_test, pipe, cv_scores, y_pred
    if 'y_proba' in locals():
        del y_proba
    if 'proba' in locals():
        del proba
    if 'cm' in locals():
        del cm
    gc.collect()
    
    return metrics_block, cm_fig, roc_fig, pr_fig


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
