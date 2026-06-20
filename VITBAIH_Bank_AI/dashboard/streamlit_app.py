"""
streamlit_app.py — Professional Banking AI Dashboard (Dark Theme).

Features:
  - Sidebar filters: age, job, housing loan
  - Probability cards with gauge
  - Real-time prediction via CatBoost model
  - Interactive charts (subscription by job, age group)
  - Metrics overview
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from catboost import CatBoostClassifier, Pool

# ─── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from preprocess import CAT_FEATURES  # noqa: E402

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VITBAIH — Bank AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme CSS
st.markdown("""
<style>
  body, .stApp { background-color: #0d1117; color: #e6edf3; }
  .block-container { padding-top: 1.5rem; }
  .metric-card {
    background: linear-gradient(135deg, #161b22, #21262d);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 6px;
  }
  .metric-value { font-size: 2.2rem; font-weight: 700; }
  .metric-label { font-size: 0.85rem; color: #8b949e; margin-top: 4px; }
  .yes-card  { border-left: 4px solid #2ea043; }
  .no-card   { border-left: 4px solid #da3633; }
  .info-card { border-left: 4px solid #388bfd; }
  section[data-testid="stSidebar"] { background: #161b22; }
  h1, h2, h3 { color: #e6edf3; }
  .stButton>button {
    background: linear-gradient(90deg, #238636, #2ea043);
    color: white; border: none; border-radius: 8px;
    padding: 0.5rem 1.5rem; font-weight: 600;
  }
  .stButton>button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)


# ─── Load artifacts ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts() -> tuple[CatBoostClassifier, list[str], pd.DataFrame]:
    model    = joblib.load(ROOT / "models" / "model.pkl")
    features = joblib.load(ROOT / "models" / "features.pkl")
    df       = pd.read_csv(ROOT / "data" / "bank-additional-full.csv", sep=";")
    return model, features, df


model, FEATURES, df = load_artifacts()


# ─── Prediction helper ────────────────────────────────────────────────────────
def predict_single(input_dict: dict) -> tuple[bool, float]:
    row = pd.DataFrame([input_dict])[FEATURES]
    for col in CAT_FEATURES:
        if col in row.columns:
            row[col] = row[col].astype(str)
    cat_idx = [FEATURES.index(c) for c in CAT_FEATURES if c in FEATURES]
    pool    = Pool(row, cat_features=cat_idx)
    prob    = float(model.predict_proba(pool)[0, 1])
    return prob >= 0.5, prob


# ─── Gauge chart ─────────────────────────────────────────────────────────────
def make_gauge(prob: float) -> go.Figure:
    color = "#2ea043" if prob >= 0.5 else "#da3633"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        delta={"reference": 50, "valueformat": ".1f"},
        number={"suffix": "%", "font": {"color": "white", "size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8b949e",
                     "tickfont": {"color": "#8b949e"}},
            "bar": {"color": color},
            "bgcolor": "#21262d",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0, 30],  "color": "#21262d"},
                {"range": [30, 50], "color": "#21262d"},
                {"range": [50, 70], "color": "#21262d"},
                {"range": [70, 100],"color": "#21262d"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "value": 50},
        },
        title={"text": "Subscription Probability", "font": {"color": "#8b949e", "size": 14}},
    ))
    fig.update_layout(
        height=250, margin=dict(l=20, r=20, t=40, b=0),
        paper_bgcolor="#0d1117", font_color="white",
    )
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 VITBAIH Bank AI")
    st.markdown("*Term Deposit Subscription Predictor*")
    st.divider()
    st.markdown("### 🔍 Customer Profile")

    age        = st.slider("Age", 18, 95, 45)
    job        = st.selectbox("Job", sorted(df["job"].unique()))
    marital    = st.selectbox("Marital Status", sorted(df["marital"].unique()))
    education  = st.selectbox("Education", sorted(df["education"].unique()))
    default    = st.selectbox("Credit Default", sorted(df["default"].unique()))
    housing    = st.selectbox("Housing Loan", sorted(df["housing"].unique()), index=0)
    loan       = st.selectbox("Personal Loan", sorted(df["loan"].unique()), index=0)
    contact    = st.selectbox("Contact Type", sorted(df["contact"].unique()))

    st.markdown("### 📞 Campaign Details")
    duration = st.slider("Call Duration (sec)", 0, 1500, 300)
    campaign = st.slider("# Contacts This Campaign", 1, 30, 2)
    month    = st.selectbox("Month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
    dow      = st.selectbox("Day of Week", ["mon","tue","wed","thu","fri"])
    poutcome = st.selectbox("Previous Outcome", sorted(df["poutcome"].unique()))
    previous = st.slider("Previous Contacts", 0, 7, 0)
    pdays    = st.slider("Days Since Last Contact", 0, 999, 999)

    st.markdown("### 📊 Economic Indicators")
    emp_var   = st.slider("Emp. Variation Rate", -3.4, 1.4, -1.8, 0.1)
    cpi       = st.slider("Consumer Price Index", 92.0, 95.0, 93.6, 0.1)
    cci       = st.slider("Consumer Conf. Index", -50.0, -26.0, -40.0, 0.5)
    euribor   = st.slider("Euribor 3M", 0.6, 5.0, 1.3, 0.1)
    nremp     = st.selectbox("Nr. Employed", [4963.6, 5008.7, 5017.5, 5023.5, 5099.1, 5176.3, 5191.0, 5195.8, 5228.1])

    predict_btn = st.button("🔮 Predict", use_container_width=True)

# ─── Main content ─────────────────────────────────────────────────────────────
st.markdown("# 🏦 Bank Marketing Intelligence Dashboard")
st.markdown("*Powered by CatBoost · VITBAIH Community Project 2026*")

# Overview metrics row
c1, c2, c3, c4 = st.columns(4)
yes_pct = (df["y"] == "yes").mean() * 100
with c1:
    st.markdown(f"""<div class="metric-card info-card">
        <div class="metric-value">{len(df):,}</div>
        <div class="metric-label">Total Customers</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card yes-card">
        <div class="metric-value">{yes_pct:.1f}%</div>
        <div class="metric-label">Subscribed (Yes)</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card no-card">
        <div class="metric-value">{100-yes_pct:.1f}%</div>
        <div class="metric-label">Not Subscribed</div></div>""", unsafe_allow_html=True)
with c4:
    imbalance = (df["y"] == "no").sum() / (df["y"] == "yes").sum()
    st.markdown(f"""<div class="metric-card info-card">
        <div class="metric-value">{imbalance:.1f}:1</div>
        <div class="metric-label">Imbalance Ratio</div></div>""", unsafe_allow_html=True)

st.divider()

# ─── Prediction section ───────────────────────────────────────────────────────
if predict_btn or "prob" in st.session_state:
    input_data = {
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "housing": housing, "loan": loan, "contact": contact,
        "month": month, "day_of_week": dow, "duration": duration, "campaign": campaign,
        "pdays": pdays, "previous": previous, "poutcome": poutcome,
        "emp.var.rate": emp_var, "cons.price.idx": cpi, "cons.conf.idx": cci,
        "euribor3m": euribor, "nr.employed": nremp,
    }
    if predict_btn:
        will_sub, prob = predict_single(input_data)
        st.session_state["prob"] = prob
        st.session_state["will_sub"] = will_sub
    else:
        prob    = st.session_state["prob"]
        will_sub = st.session_state["will_sub"]

    pred_col, gauge_col = st.columns([1, 1])
    with pred_col:
        verdict_color = "#2ea043" if will_sub else "#da3633"
        verdict_text  = "✅ WILL SUBSCRIBE" if will_sub else "❌ WILL NOT SUBSCRIBE"
        st.markdown(f"""
        <div class="metric-card" style="border-left:4px solid {verdict_color}; margin-top:10px;">
            <div class="metric-value" style="color:{verdict_color}; font-size:1.6rem">{verdict_text}</div>
            <div class="metric-label" style="margin-top:12px">Probability: <b>{prob*100:.1f}%</b></div>
            <div class="metric-label">Confidence: <b>{abs(prob-0.5)*200:.1f}%</b></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### 🔑 Key Factors")
        if duration > 300:
            st.success(f"✓ Long call duration ({duration}s)")
        if poutcome == "success":
            st.success("✓ Previous campaign successful")
        if housing == "no" and loan == "no":
            st.success("✓ No existing loans")
        if contact == "cellular":
            st.info("✓ Cellular contact (higher response)")
        if duration < 100:
            st.warning("⚠ Short call — low engagement")
        if campaign > 10:
            st.error("✗ High contact count — fatigue risk")

    with gauge_col:
        st.plotly_chart(make_gauge(prob), use_container_width=True)

st.divider()

# ─── Charts ───────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### 📊 Subscription Rate by Job")
    job_rate = (df.groupby("job")["y"]
                .apply(lambda x: (x == "yes").mean() * 100)
                .sort_values(ascending=True)
                .reset_index())
    job_rate.columns = ["job", "subscription_rate"]
    fig = px.bar(job_rate, x="subscription_rate", y="job", orientation="h",
                 color="subscription_rate", color_continuous_scale="Viridis",
                 labels={"subscription_rate": "Subscription Rate (%)", "job": ""},
                 template="plotly_dark")
    fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                      coloraxis_showscale=False, height=350)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown("#### 📊 Subscription Rate by Age Group")
    df2 = df.copy()
    bins   = [0, 25, 35, 45, 55, 65, 100]
    labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
    df2["age_group"] = pd.cut(df2["age"], bins=bins, labels=labels, right=False)
    age_rate = (df2.groupby("age_group")["y"]
                .apply(lambda x: (x == "yes").mean() * 100)
                .reset_index())
    age_rate.columns = ["age_group", "subscription_rate"]
    fig2 = px.bar(age_rate, x="age_group", y="subscription_rate",
                  color="subscription_rate", color_continuous_scale="Teal",
                  labels={"subscription_rate": "Subscription Rate (%)", "age_group": "Age Group"},
                  template="plotly_dark")
    fig2.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                       coloraxis_showscale=False, height=350)
    st.plotly_chart(fig2, use_container_width=True)

col_c, col_d = st.columns(2)
with col_c:
    st.markdown("#### 🕐 Subscription Rate by Month")
    month_order = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    month_rate  = (df.groupby("month")["y"]
                   .apply(lambda x: (x == "yes").mean() * 100)
                   .reindex(month_order)
                   .reset_index())
    month_rate.columns = ["month", "rate"]
    fig3 = px.line(month_rate, x="month", y="rate", markers=True,
                   labels={"rate": "Subscription Rate (%)", "month": "Month"},
                   template="plotly_dark", color_discrete_sequence=["#2ea043"])
    fig3.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", height=320)
    st.plotly_chart(fig3, use_container_width=True)

with col_d:
    st.markdown("#### 🗓 Target Distribution")
    tgt = df["y"].value_counts().reset_index()
    tgt.columns = ["label", "count"]
    fig4 = px.pie(tgt, names="label", values="count",
                  color="label", color_discrete_map={"yes": "#2ea043", "no": "#da3633"},
                  template="plotly_dark", hole=0.4)
    fig4.update_layout(paper_bgcolor="#0d1117", height=320)
    st.plotly_chart(fig4, use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#8b949e; font-size:0.8rem'>
VITBAIH Community Project 2026 · CatBoost · FastAPI · Streamlit · Built with ❤️
</div>""", unsafe_allow_html=True)
