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
import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os

import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State, dash_table, callback_context, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

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

CV_METRICS = [
    ("F1 Macro", "f1_macro"),
    ("Accuracy", "accuracy"),
    ("Recall Macro", "recall_macro"),
]
BINARY_ONLY_METRICS = [
    ("ROC AUC (binary)", "roc_auc"),
    ("Average Precision (binary)", "average_precision"),
]

# -----------------------
# Utilities
# -----------------------
def shorten_label(s: str, max_len: int = 18) -> str:
    s = str(s)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def df_from_store(df_json):
    if not df_json:
        return None
    if isinstance(df_json, dict):  # orient="split" dict
        return pd.DataFrame(df_json["data"], columns=df_json["columns"], index=df_json["index"])
    return pd.read_json(df_json, orient="split")

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
    _, content_string = contents.split(",")
    approx_bytes = len(content_string) * 3 // 4
    if approx_bytes > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"File too large (~{approx_bytes // (1024 * 1024):.0f} MB). "
            f"Maximum upload size is {MAX_UPLOAD_MB} MB."
        )
    decoded = base64.b64decode(content_string)

    if filename.lower().endswith(".csv"):
        return pd.read_csv(io.StringIO(decoded.decode("utf-8", errors="ignore")))
    if filename.lower().endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(decoded))
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
        template="plotly_dark",
    )
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
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
        title="ROC Curve",
        template="plotly_dark",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"PR AUC={pr_auc:.3f}, AP={ap:.3f}"))
    fig_pr.update_layout(
        title="Precision-Recall Curve",
        template="plotly_dark",
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
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title="Missingness Matrix")
        return fig

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
            colorscale=[[0.0, "black"], [1.0, "white"]],
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
        template="plotly_dark",
        title=f"Missingness Matrix (rows={sample_n:,}, cols={len(cols):,}, seed={seed})",
        margin=dict(l=40, r=40, t=60, b=120),
        height=520,
    )
    return fig

# -----------------------
# App + CSS
# -----------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Preflight"

app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
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
        dcc.Store(id="store-df-json"),
        dcc.Store(id="store-target"),
        dcc.Store(id="store-feature-types"),  # {col: type}
        dcc.Store(id="store-eda-page", data=0),  # EDA pagination state
        dcc.Store(id="store-upload-note"),
    ]
)

# -----------------------
# Panes (always mounted)
# -----------------------
health_pane = html.Div(
    id="pane-health",
    children=[
        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(html.Div(id="health-rows"), md=3),
                            dbc.Col(html.Div(id="health-cols"), md=3),
                            dbc.Col(html.Div(id="health-dups"), md=3),
                            dbc.Col(html.Div(id="health-miss-any"), md=3),
                        ],
                        className="mb-3",
                    ),
                    html.H5("Column Summary"),
                    dash_table.DataTable(
                        id="health-profile-table",
                        data=[],
                        columns=[
                            {"name": "column", "id": "column"},
                            {"name": "dtype", "id": "dtype"},
                            {"name": "missing_%", "id": "missing_%"},
                            {"name": "n_unique", "id": "n_unique"},
                            {"name": "unique_ratio", "id": "unique_ratio"},
                        ],
                        page_size=12,
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "fontFamily": "Arial",
                            "fontSize": 12,
                            "padding": "6px",
                            "backgroundColor": "#1f1f1f",
                            "color": "white",
                            "border": "1px solid #333",
                        },
                        style_header={
                            "fontWeight": "bold",
                            "backgroundColor": "#2b2b2b",
                            "border": "1px solid #444",
                        },
                        css=[
                            {
                                "selector": ".dash-table-container .previous-next-container span",
                                "rule": "color: #ffffff !important;",
                            },
                            {
                                "selector": ".dash-table-container .previous-next-container .page-number",
                                "rule": "color: #ffffff !important;",
                            },
                            {
                                "selector": ".dash-table-container .previous-next-container input",
                                "rule": "color: #ffffff !important; background-color: #2b2b2b !important;",
                            },
                            {
                                "selector": ".dash-table-container .previous-next-container input[type='text']",
                                "rule": "-webkit-text-fill-color: #ffffff !important;",
                            },
                        ],
                    ),
                    html.Div(style={"height": "12px"}),
                    html.Label("Select target column"),
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
                    html.H5("Missingness"),
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
                    dcc.Graph(id="missing-matrix-fig"),
                    dcc.Graph(id="missing-bar-fig"),
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
                                    html.Label("Low-cardinality threshold (categorical if nunique <= K)"),
                                    dcc.Slider(
                                        id="k-threshold",
                                        min=5,
                                        max=50,
                                        step=1,
                                        value=20,
                                        marks={v: str(v) for v in range(5, 51, 5)},
                                    ),
                                    html.Div(style={"height": "12px"}),
                                    html.Label("ID-like unique ratio threshold (drop if unique_ratio >= T)"),
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
                                        options=[{"label": "Treat small-unique integer columns as categorical", "value": "on"}],
                                        value=["on"],
                                    ),
                                    html.Button("Recompute", id="btn-recompute", n_clicks=0),
                                    html.Div(id="typing-need-target", style={"marginTop": "12px"}),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    html.H5("Typing Table (edit overrides)"),
                                    dash_table.DataTable(
                                        id="typing-table",
                                        data=[],
                                        columns=[
                                            {"name": "column", "id": "column", "editable": False},
                                            {"name": "suggested_type", "id": "suggested_type", "editable": False},
                                            {"name": "override_type", "id": "override_type", "presentation": "dropdown", "editable": True},
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
                                            "fontFamily": "Arial",
                                            "fontSize": 12,
                                            "padding": "6px",
                                            "backgroundColor": "#1f1f1f",
                                            "color": "white",
                                            "border": "1px solid #333",
                                            "overflow": "visible",
                                        },
                                        style_data={"overflow": "visible"},
                                        style_header={"fontWeight": "bold", "backgroundColor": "#2b2b2b", "border": "1px solid #444"},
                                        css=[
                                            {"selector": ".dash-spreadsheet-container .dropdown", "rule": "position: static !important;"},
                                            {"selector": ".dash-spreadsheet-container .Select-menu-outer", "rule": "display: block !important; z-index: 9999 !important;"},
                                            {"selector": ".dash-spreadsheet-container .Select-control", "rule": "background-color: #2b2b2b !important; color: #fff !important; border: 1px solid #666 !important;"},
                                            {"selector": ".dash-spreadsheet-container .Select-value-label", "rule": "color: #fff !important;"},
                                            {"selector": ".dash-spreadsheet-container .Select-menu-outer *", "rule": "color: #fff !important;"},
                                        ],
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(html.Button("Save Types", id="btn-save-types", n_clicks=0), width="auto"),
                                            dbc.Col(html.Div(id="typing-save-note"), width=True),
                                        ],
                                        align="center",
                                        className="mt-2",
                                    ),
                                ],
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
                                    html.Label("Pick feature(s)"),
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
                            dbc.Col(html.Button("Prev", id="btn-eda-prev", n_clicks=0), width="auto"),
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
                    html.Div(
                        id="corr-block",
                        children=[
                            html.Hr(),
                            html.H5("Correlation (numeric only)"),
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
                            dcc.Graph(id="corr-heatmap"),
                            html.Div(id="corr-scatter-container"),
                        ],
                    ),
                    html.Div(id="eda-multi-container"),
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
                        children=[
                            html.Div(id="model-metrics", className="mb-3"),
                            dbc.Row(
                                [
                                    dbc.Col(dcc.Graph(id="cm-graph"), md=6),
                                    dbc.Col(dcc.Graph(id="roc-graph"), md=6),
                                ]
                            ),
                            html.Div(style={"height": "12px"}),
                            dcc.Graph(id="pr-graph"),
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
        html.Nav(
            [
                html.A("Preflight", href="#", className="pf-logo"),
                html.Div(
                    html.A(
                        "← kooexperience.com",
                        href="https://kooexperience.com",
                        target="_blank",
                        rel="noopener",
                    ),
                    className="pf-nav-links",
                ),
            ],
            className="pf-nav",
        ),
        dbc.Row(
            dbc.Col(
                dcc.Upload(
                    id="upload-data",
                    children=html.Div([
                        "Drag and drop or ",
                        html.A("browse"),
                        html.Span(
                            f" · CSV or Parquet · max {MAX_UPLOAD_MB} MB",
                            style={"color": "var(--muted)", "fontSize": "13px"},
                        ),
                    ]),
                    style={
                        "width": "100%",
                        "height": "64px",
                        "lineHeight": "64px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "12px",
                        "textAlign": "center",
                    },
                    max_size=MAX_UPLOAD_BYTES,
                    multiple=False,
                ),
                width=12,
            ),
            className="mb-3",
        ),
        html.Div(id="upload-note", className="mt-2"),
        dbc.Row(
            dbc.Col(
                [
                    html.Label("Missing tokens (comma-separated)"),
                    dcc.Input(
                        id="missing-tokens",
                        type="text",
                        value=",".join(MISSING_TOKENS_DEFAULT),
                        style={"width": "100%"},
                    ),
                ],
                width=12,
            ),
            className="mb-3",
        ),
        html.Hr(),
        store_block,
        dcc.Tabs(
            id="tabs",
            value="tab-health",
            children=[
                dcc.Tab(label="1) Data Health", value="tab-health"),
                dcc.Tab(label="2) Missingness", value="tab-missing"),
                dcc.Tab(label="3) Data Types", value="tab-typing"),
                dcc.Tab(label="4) EDA", value="tab-eda"),
                dcc.Tab(label="5) Model", value="tab-model"),
            ],
        ),
        html.Div([health_pane, missing_pane, typing_pane, eda_pane, model_pane], style={"paddingTop": "12px"}),
    ],
)

# -----------------------
# Show/hide panes
# -----------------------
@app.callback(
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
# Upload: store df json + reset target
# (DO NOT write store-feature-types here. One writer only.)
# -----------------------
@app.callback(
    Output("store-df-json", "data"),
    Output("target-dropdown", "value"),  # reset UI, NOT the store
    Output("store-upload-note", "data"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("missing-tokens", "value"),
)
def on_upload(contents, filename, missing_tokens_str):
    if not contents or not filename:
        return None, None, None

    try:
        df = parse_contents(contents, filename)
    except ValueError as e:
        return None, None, ("error", str(e))
    except Exception as e:
        return None, None, ("error", f"Could not read file: {e}")

    tokens = [t.strip() for t in (missing_tokens_str or "").split(",")]
    df = normalize_missing_tokens(df, tokens)

    dup_count = len(df.columns) - len(set(df.columns))
    df = make_unique_columns(df)

    notes = []
    if dup_count > 0:
        notes.append(f"{dup_count} duplicate column name(s) auto-renamed.")
    if len(df) > MAX_ROWS_WARN:
        notes.append(
            f"{len(df):,} rows loaded — large datasets may be slow. "
            f"Consider a sampled subset for EDA."
        )

    note = ("warning", " ".join(notes)) if notes else None
    return df.to_json(date_format="iso", orient="split"), None, note

@app.callback(
    Output("upload-note", "children"),
    Input("store-upload-note", "data"),
)
def show_upload_note(note):
    if not note:
        return ""
    level, msg = note
    color = "danger" if level == "error" else "warning"
    return dbc.Alert(msg, color=color, dismissable=True)

# -----------------------
# Data Health: profile + dropdown options
# -----------------------
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
        return "", "", "", "", [], []

    prof = basic_profile(df)
    n_rows, n_cols = df.shape
    dup = int(df.duplicated().sum())
    miss_any = float(df.isna().any(axis=1).mean() * 100)

    return (
        f"Rows: {n_rows:,}",
        f"Columns: {n_cols:,}",
        f"Duplicate rows: {dup:,}",
        f"Rows with any missing: {miss_any:.1f}%",
        prof.to_dict("records"),
        [{"label": c, "value": c} for c in df.columns],
    )

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
            return None, "Upload data first."
        if not target:
            return None, "Select a target first."
        if not typing_rows:
            return None, "Data Types table is empty."
        
        types = {
            r["column"]: (r.get("override_type") or r.get("suggested_type") or "drop")
            for r in typing_rows
        }
        # types = {r["column"]: r.get("override_type", r.get("suggested_type", "drop")) for r in typing_rows}
        return types, f"Saved {len(types)} feature type decisions."

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
        return go.Figure(), go.Figure(), ""

    seed = int(seed or 42)
    sample_n = int(sample_n or min(5000, len(df)))
    top_n = int(top_n or 35)

    matrix_fig = missingness_heatmap_figure(df, sample_n, seed, col_mode or "all", top_n)

    miss_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    miss_pct = miss_pct[miss_pct > 0].head(25)

    if len(miss_pct) == 0:
        bar_fig = go.Figure()
        bar_fig.update_layout(template="plotly_dark", title="Top Missing Columns (up to 25)")
    else:
        bar_fig = px.bar(
            x=miss_pct.values,
            y=miss_pct.index,
            orientation="h",
            labels={"x": "Missing %", "y": "Column"},
            title="Top Missing Columns (up to 25)",
            template="plotly_dark",
        )
        bar_fig.update_yaxes(categoryorder="total ascending")
        bar_fig.update_layout(margin=dict(l=220, r=40, t=60, b=40))

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
        return [], "Upload data first."
    if not target:
        return [], "Select a target in Data Health first."

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

    label = f"Showing {start}-{end} of {total} features | Page {page + 1} of {max_page + 1}"
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
        return html.Div("Upload data first.")
    if not target:
        return html.Div("Select a target in Data Health first.")
    
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
            fig_sc = px.scatter(df, x=x, y=y, color=target, template="plotly_dark")
            fig_sc.update_layout(title=f"Scatter: {x} vs {y} by {target}", margin=dict(l=40, r=40, t=60, b=40))
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
            fig = px.histogram(df, x=feat, color=target, barmode="overlay", marginal="box", template="plotly_dark")
            fig.update_layout(title=f"{feat} distribution by {target}", margin=dict(l=40, r=40, t=60, b=40))
            per_feat_cols.append(dbc.Col(dcc.Graph(figure=fig), md=6))
        else:
            tmp = df[[feat, target]].dropna()
            if tmp.empty:
                fig = go.Figure()
                fig.update_layout(template="plotly_dark", title=f"{feat}: no non-null rows")
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
                    template="plotly_dark",
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
                    template="plotly_dark",
                    title=f"{feat} rate by category",
                    hover_data={feat: True, "feat_short": False, "rate": True, target: True},
                )

            fig.update_layout(margin=dict(l=40, r=40, t=60, b=140))
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
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title="Correlation (hidden)")
        return fig, []

    df = df_from_store(df_json)
    if df is None or not target:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title="Correlation Matrix")
        return fig, []

    feature_types = feature_types or {}

    # Apply overrides (keeps your EDA behavior consistent)
    df = apply_type_overrides(df, feature_types)

    # Type-driven inclusion: only numeric (and ordinal, if you want)
    include_types = {"numeric", "ordinal"}  # remove "ordinal" if you want it excluded
    corr_cols = [c for c, t in feature_types.items() if t in include_types and c in df.columns]

    # Safety: in case feature_types misses some cols, do not auto-include them
    # If you want fallback behavior, tell me, but this keeps it strict.

    if len(corr_cols) < 2:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title="Correlation Matrix (need ≥ 2 numeric/ordinal features)")
        return fig, []

    # Ensure columns are numeric for correlation computation
    num_df = df[corr_cols].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    if num_df.shape[1] < 2:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title="Correlation Matrix (need ≥ 2 usable numeric columns)")
        return fig, []

    method = method or "spearman"
    corr = num_df.corr(method=method)

    fig_hm = px.imshow(
        corr,
        aspect="auto",
        template="plotly_dark",
        title=f"{method.title()} correlation (typed numeric only)",
    )
    fig_hm.update_layout(margin=dict(l=40, r=40, t=60, b=40))

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
        fig_sc = px.scatter(tmp, x=a, y=b, color=target, template="plotly_dark")
        fig_sc.update_layout(title=f"{a} vs {b} | r={r:.3f}", margin=dict(l=40, r=40, t=60, b=40))
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
        return [], [], "Upload data first."
    if not target:
        return [], [], "Select a target first."
    if not feature_types:
        return [], [], "Go to Data Types and click Save Types before modeling."

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
    df = df_from_store(df_json)
    if df is None:
        return html.Div("Upload data first."), go.Figure(), go.Figure(), go.Figure()
    if not target:
        return html.Div("Select a target first."), go.Figure(), go.Figure(), go.Figure()
    if not feature_types:
        return html.Div("Save feature types in Typing tab first."), go.Figure(), go.Figure(), go.Figure()

    y_raw = df[target]
    X = df.drop(columns=[target])

    keep_cols = [c for c in X.columns if feature_types.get(c, "drop") != "drop"]
    include_features = include_features or keep_cols
    keep_cols = [c for c in keep_cols if c in set(include_features)]
    if len(keep_cols) == 0:
        return html.Div("No features selected. Use Select all or pick some features."), go.Figure(), go.Figure(), go.Figure()
    X = X[keep_cols]

    mask = y_raw.notna()
    X = X.loc[mask]
    y_raw = y_raw.loc[mask]

    train_cap = int(train_cap or 0)
    if train_cap > 0 and len(X) > train_cap:
        X = X.sample(train_cap, random_state=RANDOM_STATE)
        y_raw = y_raw.loc[X.index]

    classes = sorted(list(pd.Series(y_raw).dropna().unique()))
    is_binary = len(classes) == 2

    if (cv_metric in {"roc_auc", "average_precision"}) and not is_binary:
        cv_metric = "f1_macro"

    if is_binary:
        class_to_int = {classes[0]: 0, classes[1]: 1}
        y = y_raw.map(class_to_int).astype(int).values
        label_names = [str(classes[0]), str(classes[1])]
    else:
        y = pd.factorize(y_raw)[0]
        label_names = [str(c) for c in classes]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=RANDOM_STATE, stratify=y
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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=cv_metric)

    pipe.fit(X_train, y_train)

    y_proba = None
    if hasattr(pipe, "predict_proba") and is_binary:
        y_proba = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= float(threshold)).astype(int)
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
                title=f"ROC Curve (OvR, macro AUC={roc_auc_macro:.3f})",
                template="plotly_dark",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                margin=dict(l=40, r=40, t=60, b=40),
            )

            # Keep PR empty for now (optional to implement later)
            pr_fig = go.Figure()
            pr_fig.update_layout(
                title="Precision-Recall Curve (multiclass not implemented)",
                template="plotly_dark",
                margin=dict(l=40, r=40, t=60, b=40),
            )

            roc_note = f"Multiclass ROC-AUC (OvR macro)={roc_auc_macro:.3f}"
    else:
        roc_fig = go.Figure()
        pr_fig = go.Figure()
        roc_note = "ROC/PR requires predict_proba."

    metrics_block = dbc.Card(
        dbc.CardBody(
            [
                html.H5("Metrics"),
                html.Div(f"Rows used: {len(X):,}"),
                html.Div(
    f"Features used: {len(keep_cols)} | Numeric: {len(num_cols)} | Ordinal: {len(ord_cols)} | Categorical: {len(cat_cols)}"
),
                html.Div(f"CV metric: {cv_metric} | mean={cv_scores.mean():.3f}, std={cv_scores.std():.3f}"),
                html.Div(roc_note),
                html.H5("Classification Report (holdout)"),
                html.Pre(rep, style={"whiteSpace": "pre-wrap"}),
            ]
        )
    )

    return metrics_block, cm_fig, roc_fig, pr_fig


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)


