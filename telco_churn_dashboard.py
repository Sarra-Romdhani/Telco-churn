# ============================================================
#  Telco Customer Churn — Dashboard Streamlit Professionnel
#  pip install streamlit pandas numpy scikit-learn plotly imbalanced-learn
#  Lancer : streamlit run telco_churn_dashboard.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble       import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model   import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.metrics        import (accuracy_score, roc_auc_score, roc_curve,
                                    confusion_matrix,
                                    precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg:          #F4F6FA;
    --card:        #FFFFFF;
    --border:      #E2E8F0;
    --text:        #0F172A;
    --muted:       #64748B;
    --navy:        #1A3352;
    --navy-lt:     #2E5177;
    --orange:      #F5A623;
    --orange-lt:   #FFB84D;
    --orange-dk:   #D4891A;
    --success:     #16A34A;
    --danger:      #DC2626;
    --shadow:      0 2px 12px rgba(15,23,42,0.08);
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background-color: var(--bg) !important; }
.main  { background-color: var(--bg) !important; }
.block-container {
    background-color: var(--bg) !important;
    padding: 1.8rem 2.5rem !important;
    max-width: 1450px;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: 1px solid rgba(255,255,255,0.10) !important;
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.75) !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }
[data-testid="stSidebar"] .stRadio label {
    color: rgba(255,255,255,0.75) !important;
    font-size: .88rem; font-weight: 500;
}
[data-testid="stSidebar"] .stSelectbox label {
    color: rgba(255,255,255,0.45) !important;
    font-size: .75rem; letter-spacing: .08em; text-transform: uppercase;
}

.sidebar-avatar {
    width: 72px; height: 72px; border-radius: 50%;
    background: rgba(255,255,255,0.10);
    border: 3px solid var(--orange);
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem; margin: 0 auto 1rem;
}
.sidebar-name {
    text-align: center; font-size: 1.05rem; font-weight: 700;
    color: #FFFFFF !important; margin-bottom: .15rem;
}
.sidebar-email {
    text-align: center; font-size: .72rem;
    color: rgba(255,255,255,0.40) !important; margin-bottom: 1.5rem;
}

/* Data source badge */
.data-badge-real {
    background: rgba(22,163,74,0.20);
    border: 1px solid rgba(22,163,74,0.50);
    border-radius: 8px;
    padding: .5rem .8rem;
    font-size: .75rem;
    color: #86efac !important;
    text-align: center;
    font-weight: 600;
    margin-bottom: .5rem;
}
.data-badge-demo {
    background: rgba(245,166,35,0.15);
    border: 1px solid rgba(245,166,35,0.40);
    border-radius: 8px;
    padding: .5rem .8rem;
    font-size: .75rem;
    color: var(--orange-lt) !important;
    text-align: center;
    font-weight: 600;
    margin-bottom: .5rem;
}

.page-title {
    font-size: 2rem; font-weight: 800; color: var(--text);
    letter-spacing: -.04em; margin-bottom: .15rem; line-height: 1;
}
.page-subtitle {
    font-size: .88rem; color: var(--muted);
    margin-bottom: 1.6rem; font-weight: 400;
}
.dash-divider {
    height: 3px;
    background: linear-gradient(90deg, var(--orange) 0%, var(--navy) 60%, transparent 100%);
    border: none; margin: .4rem 0 1.6rem; border-radius: 2px;
}

.kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    box-shadow: var(--shadow);
    position: relative; overflow: hidden;
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0;
    width: 4px; height: 100%; border-radius: 4px 0 0 4px;
    background: var(--navy);
}
.kpi-card.orange-accent::before { background: var(--orange); }
.kpi-card.success-accent::before { background: var(--success); }
.kpi-card.danger-accent::before  { background: var(--danger); }

.kpi-icon {
    font-size: 1.25rem; margin-bottom: .6rem;
    display: inline-block; padding: .4rem .5rem;
    border-radius: 10px; background: var(--bg);
}
.kpi-label {
    font-size: .68rem; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: var(--muted); margin-bottom: .4rem;
}
.kpi-value {
    font-size: 2.1rem; font-weight: 800; color: var(--text); line-height: 1;
}
.kpi-delta {
    font-size: .78rem; font-weight: 600; margin-top: .5rem;
    display: inline-flex; align-items: center; gap: .3rem;
}
.delta-up      { color: var(--success); }
.delta-down    { color: var(--danger); }
.delta-neutral { color: var(--orange-dk); }

.kpi-card.full-orange {
    background: linear-gradient(135deg, var(--orange-dk), var(--orange-lt));
    border: none;
}
.kpi-card.full-orange::before { display: none; }
.kpi-card.full-orange .kpi-label,
.kpi-card.full-orange .kpi-value,
.kpi-card.full-orange .kpi-delta { color: var(--navy) !important; }
.kpi-card.full-orange .kpi-icon { background: rgba(0,0,0,0.12); }

.section-header {
    font-size: 1rem; font-weight: 700; color: var(--text);
    padding: .5rem 0 .7rem;
    border-bottom: 2px solid var(--border);
    margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: .5rem;
}
.sh-badge {
    background: var(--orange); color: var(--navy);
    font-size: .6rem; font-weight: 800; letter-spacing: .08em;
    text-transform: uppercase; padding: .15rem .55rem;
    border-radius: 20px; margin-left: auto;
}
.sh-badge-green {
    background: var(--success); color: #fff;
    font-size: .6rem; font-weight: 800; letter-spacing: .08em;
    text-transform: uppercase; padding: .15rem .55rem;
    border-radius: 20px; margin-left: auto;
}

.best-banner {
    background: var(--navy);
    border-left: 4px solid var(--orange);
    border-radius: 14px;
    padding: 1.1rem 1.5rem; margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 1rem;
}
.best-banner .trophy { font-size: 2rem; }
.best-banner .bm-title {
    font-weight: 800; font-size: 1.05rem; color: #FFFFFF;
}
.best-banner .bm-stats {
    font-size: .82rem; color: rgba(255,255,255,0.65); margin-top: .2rem;
}
.best-banner .bm-stats span { color: var(--orange); font-weight: 700; }

/* Data source info banner */
.data-info-banner {
    background: linear-gradient(135deg, #0f2740, #1A3352);
    border: 1px solid rgba(245,166,35,0.30);
    border-radius: 14px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.4rem;
    display: flex; align-items: center; gap: 1rem;
}
.data-info-banner .di-icon { font-size: 1.8rem; }
.data-info-banner .di-title {
    font-weight: 700; font-size: .95rem; color: #FFFFFF;
}
.data-info-banner .di-sub {
    font-size: .78rem; color: rgba(255,255,255,0.55); margin-top: .2rem;
}
.data-info-banner .di-sub span { color: var(--orange); font-weight: 600; }

.pred-churn {
    background: #FEF2F2;
    border: 2px solid #FECACA;
    border-radius: 16px; padding: 1.8rem; text-align: center;
}
.pred-no-churn {
    background: #F0FDF4;
    border: 2px solid #BBF7D0;
    border-radius: 16px; padding: 1.8rem; text-align: center;
}
.pred-label {
    font-size: 1rem; font-weight: 700; margin-bottom: .5rem;
    text-transform: uppercase; letter-spacing: .08em;
}
.pred-proba {
    font-size: 3rem; font-weight: 900; line-height: 1;
}
.pred-model-tag {
    display: inline-block; background: rgba(0,0,0,0.06);
    border-radius: 20px; padding: .3rem .9rem;
    font-size: .75rem; color: var(--muted); margin-top: .8rem; font-weight: 500;
}

.stButton > button {
    background: var(--navy) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: .7rem 2rem !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: .95rem !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    background: var(--orange) !important;
    color: var(--navy) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(245,166,35,.30) !important;
}

#MainMenu, footer { visibility: hidden; }

/* ── Kill the top header bar entirely ───────────────────────── */
[data-testid="stAppHeader"],
[data-testid="stAppHeader"] *,
header[data-testid="stAppHeader"],
.stAppHeader,
div.stAppHeader {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    min-height: 0 !important;
    max-height: 0 !important;
    overflow: hidden !important;
    background: transparent !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PLOTLY THEME HELPERS
# ─────────────────────────────────────────────────────────────
PALETTE   = ["#1A3352", "#F5A623", "#16A34A", "#DC2626", "#7C3AED"]
PLOT_BG   = "rgba(0,0,0,0)"
PAPER_BG  = "rgba(0,0,0,0)"
GRID_CLR  = "#E2E8F0"
FONT_CLR  = "#0F172A"
MUTED_CLR = "#64748B"
AXIS_CLR  = "#CBD5E1"

def _base_layout(**kwargs):
    base = dict(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor="#FFFFFF",
        font=dict(family="Outfit", color=FONT_CLR),
        margin=dict(t=20, b=40, l=55, r=20),
    )
    base.update(kwargs)
    return base

def _axis(title="", angle=0):
    return dict(
        title=title, gridcolor=GRID_CLR, linecolor=AXIS_CLR,
        color=MUTED_CLR, tickangle=angle,
        title_font=dict(color=MUTED_CLR, size=11),
    )

# ─────────────────────────────────────────────────────────────
# HELPERS — data
# ─────────────────────────────────────────────────────────────
def normalize(series):
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn) if mx != mn else series * 0

def encode_dataframe(df):
    """Encode raw Telco CSV (string columns) to numeric. Safe to call on already-encoded data."""
    d = df.copy()
    for col in ["Dependents", "PhoneService"]:
        if col in d.columns and d[col].dtype == object:
            d[col] = (d[col] == "Yes").astype(int)
    if "Churn" in d.columns and d["Churn"].dtype == object:
        d["Churn"] = (d["Churn"] == "Yes").astype(int)
    if "gender" in d.columns and d["gender"].dtype == object:
        d["gender"] = (d["gender"] == "Male").astype(int)
    if "SeniorCitizen" in d.columns:
        d["SeniorCitizen"] = d["SeniorCitizen"].astype(int)
    ml_map = {"No phone service": 0, "No": 1, "Yes": 2}
    if "MultipleLines" in d.columns and d["MultipleLines"].dtype == object:
        d["MultipleLines"] = d["MultipleLines"].map(ml_map).fillna(0).astype(int)
    is_map = {"No": 0, "DSL": 1, "Fiber optic": 2}
    if "InternetService" in d.columns and d["InternetService"].dtype == object:
        d["InternetService"] = d["InternetService"].map(is_map).fillna(0).astype(int)
    ct_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    if "Contract" in d.columns and d["Contract"].dtype == object:
        d["Contract"] = d["Contract"].map(ct_map).fillna(0).astype(int)
    return d

def safe_normalize(df):
    """Normalize only numeric columns that look un-normalized (max > 1.5)."""
    d = df.copy()
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in d.columns and d[col].max() > 1.5:
            d[col] = normalize(d[col].astype(float))
    return d

FEATURES = ["tenure", "Contract", "MonthlyCharges", "TotalCharges",
            "gender", "Dependents", "SeniorCitizen", "PhoneService",
            "MultipleLines", "InternetService"]

@st.cache_data(show_spinner=False)
def generate_demo_data(n=7043):
    np.random.seed(42)
    tenure     = np.random.randint(0, 73, n)
    contract   = np.random.choice([0, 1, 2], n, p=[.55, .21, .24])
    monthly    = np.round(np.random.uniform(20, 110, n), 2)
    total      = np.round(tenure * monthly + np.random.normal(0, 50, n), 2).clip(0)
    gender     = np.random.randint(0, 2, n)
    dependents = np.random.randint(0, 2, n)
    senior     = (np.random.rand(n) < .16).astype(int)
    phone      = np.random.randint(0, 2, n)
    multi      = np.random.choice([0, 1, 2], n)
    internet   = np.random.choice([0, 1, 2], n, p=[.21, .34, .45])
    logit = (-2.5 + 0.03 * monthly - 0.04 * tenure
             + 0.8 * (contract == 0) + 0.4 * (internet == 2) + 0.2 * senior)
    prob  = 1 / (1 + np.exp(-logit))
    churn = (np.random.rand(n) < prob).astype(int)
    df = pd.DataFrame(dict(
        tenure=tenure, Contract=contract,
        MonthlyCharges=monthly, TotalCharges=total,
        gender=gender, Dependents=dependents,
        SeniorCitizen=senior, PhoneService=phone,
        MultipleLines=multi, InternetService=internet,
        Churn=churn
    ))
    df["tenure"]         = normalize(df["tenure"].astype(float))
    df["MonthlyCharges"] = normalize(df["MonthlyCharges"].astype(float))
    df["TotalCharges"]   = normalize(df["TotalCharges"].astype(float))
    return df

# ─────────────────────────────────────────────────────────────
# ML helpers — now accept separate train/test sets
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_all_split(_train_df, _test_df):
    """Train on _train_df, evaluate on _test_df."""
    X_tr = _train_df[FEATURES]; y_tr = _train_df["Churn"]
    X_te = _test_df[FEATURES];  y_te = _test_df["Churn"]

    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=10,
                                                      class_weight="balanced",
                                                      random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150,
                                                          learning_rate=.1, max_depth=4,
                                                          subsample=.8, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000,
                                                  class_weight="balanced", random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=6,
                                                      class_weight="balanced", random_state=42),
        "KNN (k=5)":           KNeighborsClassifier(n_neighbors=5),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        yp   = m.predict(X_te)
        yprb = m.predict_proba(X_te)[:, 1]
        results[name] = dict(
            model=m, y_pred=yp, y_prob=yprb,
            cm=confusion_matrix(y_te, yp),
            accuracy=accuracy_score(y_te, yp),
            auc=roc_auc_score(y_te, yprb),
            precision=precision_score(y_te, yp, zero_division=0),
            recall=recall_score(y_te, yp, zero_division=0),
            f1=f1_score(y_te, yp, zero_division=0),
        )
    return results, X_tr, X_te, y_tr, y_te

@st.cache_resource(show_spinner=False)
def train_all_single(_df):
    """Fallback: single dataset → auto split 80/20."""
    X = _df[FEATURES]; y = _df["Churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.20,
                                               random_state=42, stratify=y)
    train_tmp = pd.concat([X_tr, y_tr], axis=1)
    test_tmp  = pd.concat([X_te, y_te], axis=1)
    return train_all_split(train_tmp, test_tmp)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-avatar">📡</div>
    <div class="sidebar-name">Telco Analytics</div>
    <div class="sidebar-email">churn-platform@telco.io</div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div style="height:1px;background:rgba(255,255,255,0.12);margin-bottom:1rem;"></div>',
        unsafe_allow_html=True
    )

    page = st.radio("Navigation", [
        "🏠  Vue d'ensemble",
        "📊  Analyse EDA",
        "🤖  Modèles ML",
        "🔮  Prédiction Client",
    ], label_visibility="collapsed")

    st.markdown(
        '<div style="height:1px;background:rgba(255,255,255,0.12);margin:1rem 0;"></div>',
        unsafe_allow_html=True
    )

    # ── Data upload section ───────────────────────────────────
    st.markdown(
        '<div style="font-size:.7rem;color:rgba(255,255,255,0.35);'
        'text-transform:uppercase;letter-spacing:.1em;margin-bottom:.7rem;font-weight:700;">'
        'Données réelles (R → ROSE)</div>',
        unsafe_allow_html=True
    )

    uploaded_train = st.file_uploader(
        "📂 train_data.csv  (équilibré — ROSE)",
        type="csv", key="train_upload",
        label_visibility="visible"
    )
    uploaded_test = st.file_uploader(
        "📂 test_data.csv  (réel, non équilibré)",
        type="csv", key="test_upload",
        label_visibility="visible"
    )

    st.markdown(
        '<div style="font-size:.68rem;color:rgba(255,255,255,0.28);margin-top:.4rem;">'
        'Importez les 2 fichiers produits par votre script R pour utiliser vos vraies données. '
        'Sans import → données démo actives.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="height:1px;background:rgba(255,255,255,0.12);margin:1rem 0;"></div>',
        unsafe_allow_html=True
    )

    # Single-file legacy upload (optional)
    st.markdown(
        '<div style="font-size:.7rem;color:rgba(255,255,255,0.25);'
        'text-transform:uppercase;letter-spacing:.1em;margin-bottom:.5rem;font-weight:700;">'
        'Ou fichier unique</div>',
        unsafe_allow_html=True
    )
    uploaded_single = st.file_uploader(
        "📂 Un seul CSV (split 80/20 auto)",
        type="csv", key="single_upload",
        label_visibility="visible"
    )

    st.markdown(
        '<div style="height:1px;background:rgba(255,255,255,0.12);margin:1rem 0;"></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='font-size:.72rem;color:rgba(255,255,255,0.25);text-align:center;'>"
        "Telco Churn Project · 2024<br>R (ROSE) + Python (sklearn)</div>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────
# DATA LOADING  — priority: train+test > single > demo
# ─────────────────────────────────────────────────────────────

def load_and_clean(file):
    """Read uploaded CSV, encode if needed, normalize if needed."""
    raw = pd.read_csv(file)
    raw = encode_dataframe(raw)   # safe: skips already-numeric columns
    raw = safe_normalize(raw)     # safe: skips already-normalized columns
    return raw[FEATURES + ["Churn"]].dropna()

data_source = "demo"

if uploaded_train is not None and uploaded_test is not None:
    # ── MODE 1 : real train + test from R ────────────────────
    train_df = load_and_clean(uploaded_train)
    test_df  = load_and_clean(uploaded_test)
    results, X_tr, X_te, y_tr, y_te = train_all_split(train_df, test_df)
    # For EDA and KPI display use the combined dataset
    df         = pd.concat([train_df, test_df], ignore_index=True)
    data_source = "real_split"
    source_label = (f"✅ Données réelles · train={len(train_df):,} "
                    f"(ROSE) · test={len(test_df):,}")

elif uploaded_single is not None:
    # ── MODE 2 : single CSV, auto split ──────────────────────
    df = load_and_clean(uploaded_single)
    results, X_tr, X_te, y_tr, y_te = train_all_single(df)
    data_source  = "real_single"
    source_label = f"✅ Données réelles · {len(df):,} lignes · split 80/20 auto"

else:
    # ── MODE 3 : demo data ───────────────────────────────────
    df = generate_demo_data()
    results, X_tr, X_te, y_tr, y_te = train_all_single(df)
    data_source  = "demo"
    source_label = "⚠️ Données démo — importez vos CSV pour utiliser vos vraies données"

# Show sidebar badge
badge_class = "data-badge-real" if data_source != "demo" else "data-badge-demo"
with st.sidebar:
    st.markdown(f'<div class="{badge_class}">{source_label}</div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# GLOBAL KPIs
# ─────────────────────────────────────────────────────────────
# For split mode: show train+test combined stats for overview
eval_df    = pd.concat([X_te, y_te], axis=1) if data_source != "demo" else df
n_clients  = len(df)
churn_rate = df["Churn"].mean() * 100
n_churners = df["Churn"].sum()
best_model = max(results, key=lambda m: results[m]["auc"])
best_auc   = results[best_model]["auc"]
avg_f1     = float(np.mean([results[m]["f1"] for m in results]))

# Train/test sizes for display
n_train = len(X_tr)
n_test  = len(X_te)

# ─────────────────────────────────────────────────────────────
# DATA SOURCE BANNER  (shown on every page)
# ─────────────────────────────────────────────────────────────
def render_data_banner():
    if data_source == "real_split":
        st.markdown(f"""
        <div class="data-info-banner">
          <span class="di-icon">🧬</span>
          <div>
            <div class="di-title">Données réelles · Pipeline R → Python actif</div>
            <div class="di-sub">
              Train (ROSE équilibré) : <span>{n_train:,} lignes</span> &nbsp;·&nbsp;
              Test (réel) : <span>{n_test:,} lignes</span> &nbsp;·&nbsp;
              Total : <span>{n_clients:,} clients</span>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
    elif data_source == "real_single":
        st.markdown(f"""
        <div class="data-info-banner">
          <span class="di-icon">📁</span>
          <div>
            <div class="di-title">Données réelles · Split 80/20 automatique</div>
            <div class="di-sub">
              Train : <span>{n_train:,} lignes</span> &nbsp;·&nbsp;
              Test : <span>{n_test:,} lignes</span> &nbsp;·&nbsp;
              Total : <span>{n_clients:,} clients</span>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.info(
            "⚠️ **Mode démo** — Importez `train_data.csv` et `test_data.csv` "
            "(produits par votre script R) dans la barre latérale pour afficher vos vraies données."
        )

# ─────────────────────────────────────────────────────────────
# PAGE: VUE D'ENSEMBLE
# ─────────────────────────────────────────────────────────────
if "Vue d'ensemble" in page:
    st.markdown('<div class="page-title">📡 Telco Customer Churn</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Tableau de bord analytique · Prédiction du churn client</div>',
        unsafe_allow_html=True
    )
    st.markdown('<hr class="dash-divider">', unsafe_allow_html=True)

    render_data_banner()

    # ── KPI row ──────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-icon">👥</div>
          <div class="kpi-label">Clients analysés</div>
          <div class="kpi-value">{n_clients:,}</div>
          <div class="kpi-delta delta-neutral">
            Train {n_train:,} · Test {n_test:,}
          </div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card full-orange">
          <div class="kpi-icon">⚠️</div>
          <div class="kpi-label">Taux de Churn</div>
          <div class="kpi-value">{churn_rate:.1f}%</div>
          <div class="kpi-delta">{int(n_churners):,} churners identifiés</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card success-accent">
          <div class="kpi-icon">🎯</div>
          <div class="kpi-label">Meilleur AUC-ROC</div>
          <div class="kpi-value">{best_auc:.3f}</div>
          <div class="kpi-delta delta-up">▲ {best_model.split()[0]}</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card orange-accent">
          <div class="kpi-icon">📈</div>
          <div class="kpi-label">F1-Score Moyen</div>
          <div class="kpi-value">{avg_f1:.3f}</div>
          <div class="kpi-delta delta-neutral">Sur {len(results)} modèles</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: donut + table ──────────────────────────────────
    c1, c2 = st.columns([1.1, 1.9])

    with c1:
        badge_txt = "RÉELLES" if data_source != "demo" else "DÉMO"
        badge_cls = "sh-badge-green" if data_source != "demo" else "sh-badge"
        st.markdown(
            f'<div class="section-header">🎯 Distribution Churn '
            f'<span class="{badge_cls}">{badge_txt}</span></div>',
            unsafe_allow_html=True
        )
        counts = df["Churn"].value_counts().reset_index()
        counts.columns = ["Churn", "Count"]
        counts["Label"] = counts["Churn"].map({0: "No Churn", 1: "Churn"})
        fig_pie = go.Figure(go.Pie(
            labels=counts["Label"], values=counts["Count"],
            hole=.60,
            marker=dict(colors=["#1A3352", "#F5A623"],
                        line=dict(color="#FFFFFF", width=3)),
            textinfo="percent+label",
            textfont=dict(family="Outfit", size=12, color="#0F172A"),
        ))
        fig_pie.update_layout(
            **_base_layout(height=250, margin=dict(t=10, b=10, l=10, r=10)),
            showlegend=False,
            annotations=[dict(
                text=f"<b>{churn_rate:.1f}%</b><br>Churn",
                x=.5, y=.5, font_size=15, showarrow=False,
                font=dict(family="Outfit", color=FONT_CLR)
            )]
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.markdown(
            '<div class="section-header">📈 Performances des Modèles '
            '<span class="sh-badge">5 MODÈLES</span></div>',
            unsafe_allow_html=True
        )
        comp = pd.DataFrame([
            {"Modèle": m,
             "Accuracy":  f"{results[m]['accuracy']:.3f}",
             "AUC-ROC":   f"{results[m]['auc']:.3f}",
             "Precision": f"{results[m]['precision']:.3f}",
             "Recall":    f"{results[m]['recall']:.3f}",
             "F1-Score":  f"{results[m]['f1']:.3f}"}
            for m in sorted(results, key=lambda x: -results[x]["auc"])
        ])
        st.dataframe(comp, use_container_width=True, hide_index=True,
                     column_config={
                         "Modèle":    st.column_config.TextColumn(width="medium"),
                         "AUC-ROC":   st.column_config.TextColumn(),
                         "Accuracy":  st.column_config.TextColumn(),
                         "Precision": st.column_config.TextColumn(),
                         "Recall":    st.column_config.TextColumn(),
                         "F1-Score":  st.column_config.TextColumn(),
                     })

    # ── Split stats (only in real split mode) ─────────────────
    if data_source == "real_split":
        st.markdown("<br>", unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns(3)
        train_churn_rate = y_tr.mean() * 100
        test_churn_rate  = y_te.mean() * 100
        with sc1:
            st.markdown(f"""
            <div class="kpi-card success-accent">
              <div class="kpi-icon">🔬</div>
              <div class="kpi-label">Train (ROSE équilibré)</div>
              <div class="kpi-value">{n_train:,}</div>
              <div class="kpi-delta delta-neutral">Churn : {train_churn_rate:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""
            <div class="kpi-card orange-accent">
              <div class="kpi-icon">🧪</div>
              <div class="kpi-label">Test (données réelles)</div>
              <div class="kpi-value">{n_test:,}</div>
              <div class="kpi-delta delta-neutral">Churn : {test_churn_rate:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with sc3:
            rose_effect = abs(train_churn_rate - test_churn_rate)
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-icon">⚖️</div>
              <div class="kpi-label">Effet ROSE</div>
              <div class="kpi-value">{rose_effect:.1f}pp</div>
              <div class="kpi-delta delta-neutral">Écart train vs test</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROC curves ───────────────────────────────────────────
    st.markdown(
        '<div class="section-header">📉 Courbes ROC — Comparaison des Modèles</div>',
        unsafe_allow_html=True
    )
    fig_roc = go.Figure()
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dot", color=AXIS_CLR, width=1.5))
    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_te, res["y_prob"])
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"{name} (AUC={res['auc']:.3f})",
            mode="lines",
            line=dict(color=PALETTE[i], width=2.5)
        ))
    fig_roc.update_layout(
        **_base_layout(height=360, margin=dict(t=10, b=60, l=55, r=20)),
        xaxis=dict(**_axis("Taux de Faux Positifs (FPR)")),
        yaxis=dict(**_axis("Taux de Vrais Positifs (TPR)")),
        legend=dict(orientation="h", y=-0.30,
                    font=dict(size=11, color=MUTED_CLR),
                    bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────
# PAGE: EDA
# ─────────────────────────────────────────────────────────────
elif "EDA" in page:
    st.markdown('<div class="page-title">📊 Analyse Exploratoire</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Distribution des features · Corrélations · Patterns de churn</div>',
        unsafe_allow_html=True
    )
    st.markdown('<hr class="dash-divider">', unsafe_allow_html=True)
    render_data_banner()

    df_plot = df.copy()
    df_plot["Churn_label"] = df_plot["Churn"].map({0: "No Churn", 1: "Churn"})
    COLOR_MAP = {"No Churn": "#1A3352", "Churn": "#F5A623"}

    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown(
            '<div class="section-header">💰 MonthlyCharges vs Churn</div>',
            unsafe_allow_html=True
        )
        fig_box = px.box(df_plot, x="Churn_label", y="MonthlyCharges",
                         color="Churn_label", color_discrete_map=COLOR_MAP,
                         points="outliers")
        fig_box.update_traces(marker=dict(opacity=0.5))
        fig_box.update_layout(
            **_base_layout(height=300),
            showlegend=False,
            xaxis=dict(**_axis("")),
            yaxis=dict(**_axis("Monthly Charges (normalisé)")),
        )
        st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

    with r1c2:
        st.markdown(
            '<div class="section-header">⏱ Tenure vs Churn</div>',
            unsafe_allow_html=True
        )
        fig_hist = px.histogram(df_plot, x="tenure", color="Churn_label",
                                barmode="overlay", nbins=30,
                                color_discrete_map=COLOR_MAP, opacity=.75)
        fig_hist.update_layout(
            **_base_layout(height=300),
            legend=dict(title="", font=dict(size=11, color=MUTED_CLR),
                        bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(**_axis("Tenure (normalisé)")),
            yaxis=dict(**_axis("Count")),
        )
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown(
            '<div class="section-header">📋 Contrat vs Churn</div>',
            unsafe_allow_html=True
        )
        ct_label_map = {0: "Month-to-month", 1: "One year", 2: "Two year"}
        df_ct = df.copy()
        df_ct["ContractLabel"] = df_ct["Contract"].map(ct_label_map)
        df_ct["ChurnLabel"]    = df_ct["Churn"].map({0: "No Churn", 1: "Churn"})
        ct_grp = df_ct.groupby(["ContractLabel", "ChurnLabel"]).size().reset_index(name="Count")
        fig_ct = px.bar(ct_grp, x="ContractLabel", y="Count", color="ChurnLabel",
                        barmode="group", color_discrete_map=COLOR_MAP)
        fig_ct.update_layout(
            **_base_layout(height=300),
            legend=dict(title="", font=dict(size=11, color=MUTED_CLR),
                        bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(**_axis("Type de contrat")),
            yaxis=dict(**_axis("Clients")),
        )
        st.plotly_chart(fig_ct, use_container_width=True, config={"displayModeBar": False})

    with r2c2:
        st.markdown(
            '<div class="section-header">🌐 Internet Service vs Churn</div>',
            unsafe_allow_html=True
        )
        is_label_map = {0: "No Internet", 1: "DSL", 2: "Fiber optic"}
        df_is = df.copy()
        df_is["ISLabel"]    = df_is["InternetService"].map(is_label_map)
        df_is["ChurnLabel"] = df_is["Churn"].map({0: "No Churn", 1: "Churn"})
        is_grp = df_is.groupby(["ISLabel", "ChurnLabel"]).size().reset_index(name="Count")
        fig_is = px.bar(is_grp, x="ISLabel", y="Count", color="ChurnLabel",
                        barmode="group", color_discrete_map=COLOR_MAP)
        fig_is.update_layout(
            **_base_layout(height=300),
            legend=dict(title="", font=dict(size=11, color=MUTED_CLR),
                        bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(**_axis("Service Internet")),
            yaxis=dict(**_axis("Clients")),
        )
        st.plotly_chart(fig_is, use_container_width=True, config={"displayModeBar": False})

    # ── Correlation heatmap ───────────────────────────────────
    st.markdown(
        '<div class="section-header">🔗 Matrice de Corrélation</div>',
        unsafe_allow_html=True
    )
    corr = df.corr(numeric_only=True)
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=[[0, "#1A3352"], [.5, "#F8FAFC"], [1, "#F5A623"]],
        zmid=0,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(size=10, family="JetBrains Mono", color=FONT_CLR),
        showscale=True,
    ))
    fig_corr.update_layout(
        **_base_layout(height=420, margin=dict(t=10, b=10, l=10, r=10)),
        xaxis=dict(tickangle=-30, color=FONT_CLR, linecolor=AXIS_CLR,
                   tickfont=dict(color=FONT_CLR, size=11, family="Outfit")),
        yaxis=dict(color=FONT_CLR, linecolor=AXIS_CLR,
                   tickfont=dict(color=FONT_CLR, size=11, family="Outfit")),
    )
    st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})

    # ── Train vs Test distribution (only in split mode) ───────
    if data_source == "real_split":
        st.markdown(
            '<div class="section-header">⚖️ Distribution Churn — Train (ROSE) vs Test (Réel)'
            '<span class="sh-badge-green">ROSE</span></div>',
            unsafe_allow_html=True
        )
        df_compare = pd.DataFrame({
            "Classe":  ["No Churn", "Churn", "No Churn", "Churn"],
            "Période": ["Train (ROSE)", "Train (ROSE)", "Test (Réel)", "Test (Réel)"],
            "Count":   [
                int((y_tr == 0).sum()), int((y_tr == 1).sum()),
                int((y_te == 0).sum()), int((y_te == 1).sum()),
            ]
        })
        fig_cmp = px.bar(df_compare, x="Classe", y="Count", color="Période",
                         barmode="group",
                         color_discrete_map={"Train (ROSE)": "#16A34A", "Test (Réel)": "#F5A623"})
        fig_cmp.update_layout(
            **_base_layout(height=300),
            legend=dict(title="", font=dict(size=11, color=MUTED_CLR),
                        bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(**_axis("Classe")),
            yaxis=dict(**_axis("Observations")),
        )
        st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────
# PAGE: MODÈLES ML
# ─────────────────────────────────────────────────────────────
elif "Modèles" in page:
    st.markdown('<div class="page-title">🤖 Évaluation des Modèles ML</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Comparaison · Matrices de confusion · Feature importance</div>',
        unsafe_allow_html=True
    )
    st.markdown('<hr class="dash-divider">', unsafe_allow_html=True)
    render_data_banner()

    bm = results[best_model]
    st.markdown(f"""
    <div class="best-banner">
      <span class="trophy">🏆</span>
      <div>
        <div class="bm-title">Meilleur modèle : {best_model}</div>
        <div class="bm-stats">
          AUC-ROC <span>{bm['auc']:.4f}</span> ·
          Accuracy <span>{bm['accuracy']:.4f}</span> ·
          F1-Score <span>{bm['f1']:.4f}</span> ·
          Recall <span>{bm['recall']:.4f}</span>
          {"· <span>Entraîné sur données ROSE</span>" if data_source == "real_split" else ""}
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Metric bar charts ─────────────────────────────────────
    metrics = ["accuracy", "auc", "precision", "recall", "f1"]
    labels  = ["Accuracy", "AUC-ROC", "Precision", "Recall", "F1-Score"]
    fig_bar = make_subplots(rows=1, cols=5, subplot_titles=labels,
                            horizontal_spacing=.05)
    for i, (met, lab) in enumerate(zip(metrics, labels)):
        vals  = [results[m][met] for m in results]
        names = list(results.keys())
        short = [n.split()[0] for n in names]
        bar_colors = [PALETTE[0] if n == best_model else "#E2E8F0" for n in names]
        fig_bar.add_trace(go.Bar(
            x=short, y=vals, name=lab,
            marker_color=bar_colors,
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
            textfont=dict(size=9, color=MUTED_CLR),
        ), row=1, col=i + 1)

    fig_bar.update_layout(
        **_base_layout(height=300, margin=dict(t=45, b=10, l=10, r=10)),
        showlegend=False,
    )
    for i in range(1, 6):
        fig_bar.update_yaxes(range=[0, 1.15], gridcolor=GRID_CLR,
                              color=MUTED_CLR, linecolor=AXIS_CLR, row=1, col=i)
        fig_bar.update_xaxes(tickfont=dict(size=8, color=MUTED_CLR),
                              linecolor=AXIS_CLR, row=1, col=i)
    for ann in fig_bar.layout.annotations:
        ann.font.color = MUTED_CLR
        ann.font.size  = 11

    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # ── Confusion matrices ────────────────────────────────────
    cm_badge = '<span class="sh-badge-green">TEST RÉEL</span>' if data_source != "demo" else ""
    st.markdown(
        f'<div class="section-header">🔢 Matrices de Confusion{cm_badge}</div>',
        unsafe_allow_html=True
    )
    cols_cm = st.columns(len(results))
    for col, (name, res) in zip(cols_cm, results.items()):
        with col:
            cm = res["cm"]
            fig_cm = go.Figure(go.Heatmap(
                z=cm, x=["No Churn", "Churn"], y=["No Churn", "Churn"],
                colorscale=[[0, "#EFF6FF"], [1, "#1A3352"]],
                text=cm, texttemplate="%{text}",
                textfont=dict(size=13, family="JetBrains Mono", color=FONT_CLR),
                showscale=False,
            ))
            fig_cm.update_layout(
                title=dict(text=name.split()[0],
                           font=dict(size=11, family="Outfit", color=MUTED_CLR)),
                **_base_layout(height=200, margin=dict(t=35, b=30, l=40, r=10)),
                xaxis=dict(title="Prédit", tickfont=dict(size=8, color=MUTED_CLR),
                           linecolor=AXIS_CLR, color=MUTED_CLR),
                yaxis=dict(title="Réel", tickfont=dict(size=8, color=MUTED_CLR),
                           linecolor=AXIS_CLR, color=MUTED_CLR),
            )
            st.plotly_chart(fig_cm, use_container_width=True,
                            config={"displayModeBar": False})

    # ── Feature importance ────────────────────────────────────
    st.markdown(
        '<div class="section-header">🌲 Feature Importance — Random Forest</div>',
        unsafe_allow_html=True
    )
    rf = results["Random Forest"]["model"]
    fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    bar_fi_colors = [PALETTE[0] if v == fi.max() else "#BFDBFE" for v in fi.values]
    fig_fi = go.Figure(go.Bar(
        x=fi.values, y=fi.index, orientation="h",
        marker_color=bar_fi_colors,
        text=[f"{v:.4f}" for v in fi.values],
        textposition="outside",
        textfont=dict(size=10, family="JetBrains Mono", color=FONT_CLR),
    ))
    fig_fi.update_layout(
        **_base_layout(height=340, margin=dict(t=10, b=30, l=120, r=80)),
        xaxis=dict(title="Importance (Gini)", gridcolor=GRID_CLR, linecolor=AXIS_CLR,
                   color=FONT_CLR, tickfont=dict(color=FONT_CLR, size=11),
                   title_font=dict(color=FONT_CLR, size=11)),
        yaxis=dict(color=FONT_CLR, linecolor=AXIS_CLR, title="",
                   tickfont=dict(color=FONT_CLR, size=11)),
    )
    st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})

    # ── Radar chart ───────────────────────────────────────────
    st.markdown(
        '<div class="section-header">🕸 Radar des Métriques</div>',
        unsafe_allow_html=True
    )
    met_keys   = ["accuracy", "auc", "precision", "recall", "f1"]
    met_labels = ["Accuracy", "AUC-ROC", "Precision", "Recall", "F1-Score"]
    fig_rad = go.Figure()
    for i, (name, res) in enumerate(results.items()):
        vals = [res[k] for k in met_keys] + [res[met_keys[0]]]
        labs = met_labels + [met_labels[0]]
        fig_rad.add_trace(go.Scatterpolar(
            r=vals, theta=labs, name=name,
            line=dict(color=PALETTE[i], width=2.5),
            fill="toself", fillcolor=PALETTE[i], opacity=.12,
        ))
    fig_rad.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor=GRID_CLR,
                            tickfont=dict(size=9, color=MUTED_CLR),
                            color=MUTED_CLR),
            angularaxis=dict(gridcolor=GRID_CLR, color=MUTED_CLR),
            bgcolor="#FFFFFF",
        ),
        paper_bgcolor=PAPER_BG,
        height=420, margin=dict(t=30, b=40, l=40, r=40),
        legend=dict(font=dict(size=11, family="Outfit", color=MUTED_CLR),
                    orientation="h", y=-0.15, bgcolor="rgba(0,0,0,0)"),
        font=dict(family="Outfit", color=FONT_CLR),
    )
    st.plotly_chart(fig_rad, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────
# PAGE: PRÉDICTION CLIENT
# ─────────────────────────────────────────────────────────────
elif "Prédiction" in page:
    st.markdown(
        "<div class='page-title'>🔮 Prédiction d'un Nouveau Client</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='page-subtitle'>Saisissez les caractéristiques du client "
        "pour prédire son risque de churn</div>",
        unsafe_allow_html=True
    )
    st.markdown('<hr class="dash-divider">', unsafe_allow_html=True)
    render_data_banner()

    sel_model = st.selectbox(
        "Modèle de prédiction", list(results.keys()), index=0,
        help="Choisissez le modèle ML à utiliser"
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown(
            '<div class="section-header">👤 Profil Client</div>',
            unsafe_allow_html=True
        )
        gender     = st.selectbox("Genre", ["Male", "Female"])
        senior     = st.selectbox("Senior Citizen", ["Non (0)", "Oui (1)"])
        dependents = st.selectbox("Dépendants", ["Non", "Oui"])
        phone_svc  = st.selectbox("Service Téléphonique", ["Non", "Oui"])

    with col_b:
        st.markdown(
            '<div class="section-header">📦 Services</div>',
            unsafe_allow_html=True
        )
        multi_lines = st.selectbox("Lignes Multiples",
                                   ["No phone service", "No", "Yes"])
        internet    = st.selectbox("Service Internet",
                                   ["No", "DSL", "Fiber optic"])
        contract    = st.selectbox("Type de Contrat",
                                   ["Month-to-month", "One year", "Two year"])

    with col_c:
        st.markdown(
            '<div class="section-header">💵 Facturation</div>',
            unsafe_allow_html=True
        )
        tenure_raw  = st.slider("Ancienneté (mois)", 0, 72, 12)
        monthly_raw = st.slider("Charges Mensuelles ($)", 20.0, 110.0, 65.0, step=0.5)
        total_raw   = st.slider("Charges Totales ($)", 0.0, 8000.0,
                                float(tenure_raw) * monthly_raw, step=10.0)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮  Prédire le Churn",
                            use_container_width=True, type="primary")

    if predict_btn:
        gender_enc   = 1 if gender == "Male" else 0
        senior_enc   = 1 if "Oui" in senior else 0
        dep_enc      = 1 if dependents == "Oui" else 0
        phone_enc    = 1 if phone_svc == "Oui" else 0
        multi_enc    = {"No phone service": 0, "No": 1, "Yes": 2}[multi_lines]
        internet_enc = {"No": 0, "DSL": 1, "Fiber optic": 2}[internet]
        contract_enc = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]

        tenure_n  = tenure_raw / 72
        monthly_n = (monthly_raw - 20) / (110 - 20)
        total_n   = min(total_raw / 8000, 1.0)

        input_vec  = np.array([[tenure_n, contract_enc, monthly_n, total_n,
                                gender_enc, dep_enc, senior_enc, phone_enc,
                                multi_enc, internet_enc]])
        model_obj  = results[sel_model]["model"]
        pred_class = model_obj.predict(input_vec)[0]
        pred_proba = model_obj.predict_proba(input_vec)[0][1]

        st.markdown("<br>", unsafe_allow_html=True)
        _, res_col, _ = st.columns([1, 2, 1])
        with res_col:
            if pred_class == 1:
                st.markdown(f"""
                <div class="pred-churn">
                  <div style="font-size:3rem;margin-bottom:.5rem">⚠️</div>
                  <div class="pred-label" style="color:#DC2626;">RISQUE DE CHURN DÉTECTÉ</div>
                  <div class="pred-proba" style="color:#DC2626;">{pred_proba*100:.1f}%</div>
                  <div class="pred-model-tag">Modèle : {sel_model}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-no-churn">
                  <div style="font-size:3rem;margin-bottom:.5rem">✅</div>
                  <div class="pred-label" style="color:#16A34A;">CLIENT FIDÈLE</div>
                  <div class="pred-proba" style="color:#16A34A;">{(1-pred_proba)*100:.1f}%</div>
                  <div class="pred-model-tag">Modèle : {sel_model}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        gauge_color = "#DC2626" if pred_proba > .5 else "#16A34A"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_proba * 100,
            delta={"reference": 50, "valueformat": ".1f", "suffix": "%",
                   "increasing": {"color": "#DC2626"},
                   "decreasing": {"color": "#16A34A"}},
            title={"text": "Probabilité de Churn (%)",
                   "font": {"size": 14, "family": "Outfit", "color": MUTED_CLR}},
            number={"suffix": "%",
                    "font": {"size": 40, "family": "Outfit", "color": gauge_color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1,
                         "tickcolor": MUTED_CLR,
                         "tickfont": {"color": MUTED_CLR}},
                "bar":  {"color": gauge_color, "thickness": .25},
                "bgcolor": "#F8FAFC",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30],   "color": "#DCFCE7"},
                    {"range": [30, 60],  "color": "#FEF9C3"},
                    {"range": [60, 100], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": "#F5A623", "width": 3},
                    "thickness": .80, "value": 50
                },
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor=PAPER_BG,
            height=300, margin=dict(t=30, b=10, l=40, r=40),
            font=dict(family="Outfit", color=FONT_CLR),
        )
        st.plotly_chart(fig_gauge, use_container_width=True,
                        config={"displayModeBar": False})

        st.markdown(
            '<div class="section-header">📋 Récapitulatif du Profil</div>',
            unsafe_allow_html=True
        )
        summary_data = {
            "Feature": ["Ancienneté", "Contrat", "Charges Mensuelles", "Charges Totales",
                        "Genre", "Dépendants", "Senior", "Tél.", "Multi-lignes", "Internet"],
            "Valeur saisie": [
                f"{tenure_raw} mois", contract,
                f"${monthly_raw:.0f}/mois", f"${total_raw:.0f}",
                gender, dependents, senior.split()[0],
                phone_svc, multi_lines, internet
            ],
            "Valeur encodée": [
                f"{tenure_n:.3f}", str(contract_enc),
                f"{monthly_n:.3f}", f"{total_n:.3f}",
                str(gender_enc), str(dep_enc),
                str(senior_enc), str(phone_enc),
                str(multi_enc), str(internet_enc)
            ]
        }
        st.dataframe(pd.DataFrame(summary_data),
                     use_container_width=True, hide_index=True)