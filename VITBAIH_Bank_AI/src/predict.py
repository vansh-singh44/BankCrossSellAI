"""
predict.py — Run predictions on 5 sample customers.
Exports outputs/predictions.csv with features, prediction, probability, confidence.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR  = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

from preprocess import CAT_FEATURES


SAMPLE_CUSTOMERS = [
    # Customer 1 — likely YES: long duration, success poutcome, cellular contact
    {
        "age": 45, "job": "management", "marital": "married", "education": "university.degree",
        "default": "no", "housing": "no", "loan": "no", "contact": "cellular",
        "month": "oct", "day_of_week": "mon", "duration": 520, "campaign": 1,
        "pdays": 3, "previous": 2, "poutcome": "success",
        "emp.var.rate": -1.8, "cons.price.idx": 93.2, "cons.conf.idx": -36.4,
        "euribor3m": 0.9, "nr.employed": 5099.1,
    },
    # Customer 2 — likely YES: retired, no loans, cellular
    {
        "age": 67, "job": "retired", "marital": "single", "education": "basic.9y",
        "default": "no", "housing": "no", "loan": "no", "contact": "cellular",
        "month": "nov", "day_of_week": "wed", "duration": 410, "campaign": 1,
        "pdays": 999, "previous": 0, "poutcome": "nonexistent",
        "emp.var.rate": -2.9, "cons.price.idx": 92.8, "cons.conf.idx": -38.3,
        "euribor3m": 1.1, "nr.employed": 5017.5,
    },
    # Customer 3 — likely NO: many campaigns, short duration, telephone
    {
        "age": 38, "job": "blue-collar", "marital": "married", "education": "basic.4y",
        "default": "no", "housing": "yes", "loan": "yes", "contact": "telephone",
        "month": "may", "day_of_week": "thu", "duration": 52, "campaign": 18,
        "pdays": 999, "previous": 0, "poutcome": "nonexistent",
        "emp.var.rate": 1.4, "cons.price.idx": 94.5, "cons.conf.idx": -45.1,
        "euribor3m": 4.8, "nr.employed": 5228.1,
    },
    # Customer 4 — likely NO: failed previous, short call
    {
        "age": 28, "job": "services", "marital": "single", "education": "high.school",
        "default": "unknown", "housing": "yes", "loan": "no", "contact": "telephone",
        "month": "may", "day_of_week": "fri", "duration": 85, "campaign": 12,
        "pdays": 999, "previous": 1, "poutcome": "failure",
        "emp.var.rate": 1.1, "cons.price.idx": 94.0, "cons.conf.idx": -42.7,
        "euribor3m": 4.6, "nr.employed": 5191.0,
    },
    # Customer 5 — borderline YES: admin, moderate duration, cellular
    {
        "age": 52, "job": "admin.", "marital": "divorced", "education": "professional.course",
        "default": "no", "housing": "no", "loan": "no", "contact": "cellular",
        "month": "aug", "day_of_week": "tue", "duration": 320, "campaign": 3,
        "pdays": 999, "previous": 0, "poutcome": "nonexistent",
        "emp.var.rate": -1.1, "cons.price.idx": 93.8, "cons.conf.idx": -39.8,
        "euribor3m": 1.5, "nr.employed": 5099.1,
    },
]


def load_artifacts() -> tuple[CatBoostClassifier, list[str]]:
    model    = joblib.load(MODELS_DIR / "model.pkl")
    features = joblib.load(MODELS_DIR / "features.pkl")
    return model, features


def predict_customers(
    customers: list[dict],
    model: CatBoostClassifier,
    feature_names: list[str],
) -> pd.DataFrame:
    """Run predictions and return formatted DataFrame."""
    df = pd.DataFrame(customers)[feature_names]
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str)

    cat_idx = [feature_names.index(c) for c in CAT_FEATURES if c in feature_names]
    pool    = Pool(df, cat_features=cat_idx)

    preds   = model.predict(pool).flatten()
    probs   = model.predict_proba(pool)[:, 1]

    results = df.copy()
    results["prediction"]    = ["YES" if p == 1 else "NO" for p in preds]
    results["probability"]   = np.round(probs, 4)
    results["confidence"]    = np.round(np.abs(probs - 0.5) * 2, 4)  # 0=uncertain, 1=certain
    results["customer_id"]   = [f"CUST_{i+1:03d}" for i in range(len(customers))]

    # Reorder
    front = ["customer_id", "prediction", "probability", "confidence"]
    cols  = front + [c for c in results.columns if c not in front]
    return results[cols]


def run_predictions() -> pd.DataFrame:
    model, features = load_artifacts()
    results = predict_customers(SAMPLE_CUSTOMERS, model, features)

    out_path = OUTPUTS_DIR / "predictions.csv"
    results.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print("PREDICTIONS")
    print("=" * 60)
    display_cols = ["customer_id", "age", "job", "duration", "poutcome",
                    "prediction", "probability", "confidence"]
    print(results[display_cols].to_string(index=False))
    print(f"\n✓ Saved → {out_path}")

    yes_count = (results["prediction"] == "YES").sum()
    no_count  = (results["prediction"] == "NO").sum()
    print(f"\nSummary: {yes_count} YES | {no_count} NO")
    return results


if __name__ == "__main__":
    run_predictions()
