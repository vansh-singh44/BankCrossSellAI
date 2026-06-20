# 🏦 VITBAIH Bank Marketing AI — Track C Submission

> **Predict which customers will subscribe to a term deposit** using CatBoost,
> served via FastAPI, and visualised through a Streamlit dashboard.

---

## 📋 Project Overview

A production-quality end-to-end ML pipeline on the UCI Bank Marketing dataset
(41,188 customers, 20 features). The pipeline covers EDA → preprocessing →
model training → evaluation → FastAPI serving → Streamlit dashboard.

```
VITBAIH_Bank_AI/
├── data/              # bank-additional-full.csv (semicolon-delimited)
├── notebooks/         # eda.ipynb (exploratory notebook)
├── models/            # Saved artefacts: model.pkl, features.pkl, encoder.pkl
├── plots/             # 10 generated plots (EDA + evaluation)
├── outputs/           # predictions.csv
├── src/
│   ├── eda.py         # Phase 1 — EDA & plots
│   ├── preprocess.py  # Phase 2 — Feature engineering
│   ├── train.py       # Phase 3 — LR baseline + CatBoost
│   ├── evaluate.py    # Phase 3 — Metrics, confusion matrix, comparison
│   └── predict.py     # Phase 4 — 5-customer predictions
├── api/
│   └── app.py         # Phase 6 — FastAPI (predict / health / explain)
├── dashboard/
│   └── streamlit_app.py  # Phase 7 — Streamlit banking UI
├── README.md
├── EXPLANATION.md
├── requirements.txt
└── .gitignore
```

---

## 🏗 Architecture

```
CSV Dataset
    │
    ▼
[src/eda.py]          ── EDA plots ──────────────────► plots/
    │
    ▼
[src/preprocess.py]   ── encode target, cat features, 80/20 split
    │
    ▼
[src/train.py]        ── LR baseline + CatBoostClassifier
    │                    models/model.pkl · lr_model.pkl · features.pkl
    ▼
[src/evaluate.py]     ── Accuracy / Precision / Recall / F1 / AUC
    │                    Confusion matrices · ROC curves · Feature importance
    ▼
[src/predict.py]      ── 5 sample predictions → outputs/predictions.csv
    │
    ├─► [api/app.py]              FastAPI  :8000
    └─► [dashboard/streamlit_app.py]  Streamlit :8501
```

---

## 📊 Dataset

| Property          | Value                    |
|-------------------|--------------------------|
| Source            | UCI Bank Marketing       |
| Rows              | 41,188                   |
| Features          | 20 (10 categorical, 10 numerical) |
| Target            | `y` — term deposit subscription (yes/no) |
| Imbalance         | ~11.3% yes / 88.7% no        |
| Missing values    | 0 (unknown encoded as category) |

---

## 🤖 Model Details

### Baseline — Logistic Regression
- OrdinalEncoder for categoricals
- StandardScaler for numerics
- `class_weight="balanced"`
- `max_iter=1000`, `solver="lbfgs"`

### Main — CatBoostClassifier
- Native categorical handling (no encoding needed)
- `auto_class_weights="Balanced"` for imbalance
- `iterations=500`, `learning_rate=0.05`, `depth=6`
- Early stopping on AUC (patience=30)

### Results

| Metric    | Logistic Regression | CatBoost  |
|-----------|--------------------:|----------:|
| Accuracy  | 0.8569              | 0.8651    |
| Precision | 0.4347              | 0.4527    |
| Recall    | 0.8998              | 0.9440    |
| F1        | 0.5862              | 0.6119    |
| ROC-AUC   | 0.9387              | 0.9566    |

**CatBoost wins on all metrics.** F1 is used as the primary metric due to class imbalance.

---

## 🚀 Installation

```bash
# 1. Clone / navigate to project
cd VITBAIH_Bank_AI

# 2. Create virtual environment (recommended)
python3 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run pipeline end-to-end
python src/eda.py        # Phase 1 — EDA
python src/train.py      # Phase 2+3 — preprocess + train
python src/evaluate.py   # Phase 3 — evaluate
python src/predict.py    # Phase 4 — predict 5 customers
```

---

## 🌐 API Usage

```bash
# Start FastAPI server
cd api && uvicorn app:app --reload --port 8000
```

### GET /health
```bash
curl http://localhost:8000/health
# {"status":"ok","model":"CatBoost"}
```

### POST /predict
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45, "job": "management", "marital": "married",
    "education": "university.degree", "default": "no",
    "housing": "no", "loan": "no", "contact": "cellular",
    "month": "oct", "day_of_week": "mon", "duration": 520,
    "campaign": 1, "pdays": 3, "previous": 2, "poutcome": "success",
    "emp.var.rate": -1.8, "cons.price.idx": 93.2,
    "cons.conf.idx": -36.4, "euribor3m": 0.9, "nr.employed": 5099.1
  }'
```
Response:
```json
{
  "will_subscribe": true,
  "probability": 0.8953,
  "confidence": 0.7906,
  "top_factors": [
    "Long call duration (520s) — strong engagement signal",
    "Previous campaign was successful",
    "Contacted via cellular — higher response rate"
  ]
}
```

### POST /explain (Groq LLM)
```bash
# Set GROQ_API_KEY env var for LLM explanations
export GROQ_API_KEY=gsk_...

curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{ ... same payload ... }'
```
Response:
```json
{
  "explanation": "Customer likely to subscribe because they have strong savings and no liabilities.",
  "recommendation": "RM should discuss term deposits and highlight current rates."
}
```

---

## 📈 Streamlit Dashboard

```bash
streamlit run dashboard/streamlit_app.py
# Opens at http://localhost:8501
```

Features:
- Sidebar: age slider, job dropdown, housing loan filter, economic indicators
- Real-time prediction with probability gauge
- Subscription rate charts by job, age group, month
- Target distribution donut chart
- Dark banking-grade UI

---

## 📁 Generated Outputs

| File | Description |
|------|-------------|
| `plots/01_target_distribution.png` | Class imbalance visual |
| `plots/02_age_distribution.png`    | Age histogram by subscription |
| `plots/03_duration_distribution.png` | Call duration distribution |
| `plots/04_job_vs_subscription.png` | Subscription rate per job |
| `plots/05_housing_vs_subscription.png` | Housing loan effect |
| `plots/06_age_group_vs_subscription.png` | Age group breakdown |
| `plots/07_correlation_heatmap.png` | Feature correlation matrix |
| `plots/08a_cm_lr.png`              | LR confusion matrix |
| `plots/08b_cm_catboost.png`        | CatBoost confusion matrix |
| `plots/08_roc_curves.png`          | ROC curve comparison |
| `plots/09_feature_importance.png`  | CatBoost feature importances |
| `plots/10_model_comparison.png`    | Side-by-side metric comparison |
| `outputs/predictions.csv`          | 5 sample customer predictions |
| `models/model.pkl`                 | CatBoost model |
| `models/lr_model.pkl`              | Logistic Regression pipeline |
| `models/features.pkl`              | Feature name list |

---

## 🔧 Environment Variables

| Variable     | Description                        | Default |
|-------------|-------------------------------------|---------|
| `GROQ_API_KEY` | Groq API key for `/explain` LLM  | (optional — falls back to rule-based) |

---

*VITBAIH Community Project 2026 · Track C · Senior ML Engineer Submission*
