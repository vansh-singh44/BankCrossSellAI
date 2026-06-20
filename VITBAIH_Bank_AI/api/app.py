"""
app.py — FastAPI service for Bank Marketing Term Deposit Prediction.

Endpoints:
  POST /predict   — CatBoost prediction with top factors
  GET  /health    — Health check
  POST /explain   — LLM explanation via Groq API (llama-3.3-70b-versatile)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import httpx
import joblib
import numpy as np
import pandas as pd
import uvicorn
from catboost import CatBoostClassifier, Pool
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import CAT_FEATURES  # noqa: E402

# ─── App init ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="VITBAIH Bank Marketing API",
    description="Predict term deposit subscription likelihood using CatBoost.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load artifacts at startup ────────────────────────────────────────────────
_model: CatBoostClassifier | None = None
_features: list[str] | None = None


def get_model() -> tuple[CatBoostClassifier, list[str]]:
    global _model, _features
    if _model is None:
        models_dir = ROOT / "models"
        _model    = joblib.load(models_dir / "model.pkl")
        _features = joblib.load(models_dir / "features.pkl")
    return _model, _features


# ─── Schemas ──────────────────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    age: int                    = Field(..., example=45)
    job: str                    = Field(..., example="management")
    marital: str                = Field(..., example="married")
    education: str              = Field(..., example="university.degree")
    default: str                = Field(..., example="no")
    housing: str                = Field(..., example="no")
    loan: str                   = Field(..., example="no")
    contact: str                = Field(..., example="cellular")
    month: str                  = Field(..., example="oct")
    day_of_week: str            = Field(..., example="mon")
    duration: int               = Field(..., example=520)
    campaign: int               = Field(..., example=1)
    pdays: int                  = Field(..., example=3)
    previous: int               = Field(..., example=2)
    poutcome: str               = Field(..., example="success")
    emp_var_rate: float         = Field(..., alias="emp.var.rate", example=-1.8)
    cons_price_idx: float       = Field(..., alias="cons.price.idx", example=93.2)
    cons_conf_idx: float        = Field(..., alias="cons.conf.idx", example=-36.4)
    euribor3m: float            = Field(..., example=0.9)
    nr_employed: float          = Field(..., alias="nr.employed", example=5099.1)

    model_config = {"populate_by_name": True}


class PredictResponse(BaseModel):
    will_subscribe: bool
    probability: float
    confidence: float
    top_factors: list[str]


class ExplainResponse(BaseModel):
    explanation: str
    recommendation: str


# ─── Helpers ──────────────────────────────────────────────────────────────────
def customer_to_df(customer: CustomerInput, feature_names: list[str]) -> pd.DataFrame:
    """Convert Pydantic input to a properly-typed DataFrame row."""
    raw = {
        "age":            customer.age,
        "job":            customer.job,
        "marital":        customer.marital,
        "education":      customer.education,
        "default":        customer.default,
        "housing":        customer.housing,
        "loan":           customer.loan,
        "contact":        customer.contact,
        "month":          customer.month,
        "day_of_week":    customer.day_of_week,
        "duration":       customer.duration,
        "campaign":       customer.campaign,
        "pdays":          customer.pdays,
        "previous":       customer.previous,
        "poutcome":       customer.poutcome,
        "emp.var.rate":   customer.emp_var_rate,
        "cons.price.idx": customer.cons_price_idx,
        "cons.conf.idx":  customer.cons_conf_idx,
        "euribor3m":      customer.euribor3m,
        "nr.employed":    customer.nr_employed,
    }
    df = pd.DataFrame([raw])[feature_names]
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def derive_top_factors(customer: CustomerInput, probability: float) -> list[str]:
    """Rule-based top-factor derivation for RM-friendly output."""
    factors: list[str] = []

    if customer.duration > 300:
        factors.append(f"Long call duration ({customer.duration}s) — strong engagement signal")
    if customer.poutcome == "success":
        factors.append("Previous campaign was successful")
    if customer.contact == "cellular":
        factors.append("Contacted via cellular — higher response rate")
    if customer.housing == "no" and customer.loan == "no":
        factors.append("No existing loans — financially flexible")
    if customer.job in ("retired", "management", "admin."):
        factors.append(f"Job category '{customer.job}' shows above-average subscription rate")
    if customer.emp_var_rate < -1.5:
        factors.append("Favourable macroeconomic environment (low emp.var.rate)")
    if customer.pdays < 30:
        factors.append(f"Recently contacted {customer.pdays} days ago — warm lead")

    if not factors:
        if probability > 0.5:
            factors = ["Moderate probability based on combined customer profile"]
        else:
            factors = ["Short call duration limits engagement signal",
                       "No prior successful contact on record"]
    return factors[:3]


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Utility"])
def health_check() -> dict[str, str]:
    """Check API and model status."""
    return {"status": "ok", "model": "CatBoost"}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(customer: CustomerInput) -> PredictResponse:
    """Predict term deposit subscription for a single customer."""
    try:
        model, features = get_model()
        df  = customer_to_df(customer, features)
        cat_idx = [features.index(c) for c in CAT_FEATURES if c in features]
        pool    = Pool(df, cat_features=cat_idx)

        prob        = float(model.predict_proba(pool)[0, 1])
        pred        = prob >= 0.5
        confidence  = round(abs(prob - 0.5) * 2, 4)
        top_factors = derive_top_factors(customer, prob)

        return PredictResponse(
            will_subscribe=pred,
            probability=round(prob, 4),
            confidence=confidence,
            top_factors=top_factors,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/explain", response_model=ExplainResponse, tags=["Explanation"])
async def explain(customer: CustomerInput) -> ExplainResponse:
    """
    Generate a plain-English RM explanation via Groq (llama-3.3-70b-versatile).
    Falls back to rule-based explanation if GROQ_API_KEY is not set.
    """
    # First get prediction
    model, features = get_model()
    df  = customer_to_df(customer, features)
    cat_idx = [features.index(c) for c in CAT_FEATURES if c in features]
    pool    = Pool(df, cat_features=cat_idx)
    prob    = float(model.predict_proba(pool)[0, 1])
    pred    = "subscribe" if prob >= 0.5 else "NOT subscribe"
    factors = derive_top_factors(customer, prob)

    groq_key = os.getenv("GROQ_API_KEY", "")

    if groq_key:
        prompt = (
            f"You are a senior relationship manager at a bank. "
            f"A customer (age {customer.age}, job: {customer.job}, marital: {customer.marital}) "
            f"has been assessed by our AI model with a {prob*100:.1f}% likelihood to subscribe "
            f"to a term deposit. Key factors: {'; '.join(factors)}. "
            f"Write a brief 2-sentence explanation for a junior RM and a one-line action recommendation. "
            f"Format: EXPLANATION: <text> | RECOMMENDATION: <text>"
        )
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_key}",
                             "Content-Type": "application/json"},
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 200,
                        "temperature": 0.4,
                    },
                )
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]
                parts = text.split("|")
                explanation    = parts[0].replace("EXPLANATION:", "").strip() if parts else text
                recommendation = parts[1].replace("RECOMMENDATION:", "").strip() if len(parts) > 1 else "Follow up within 48 hours."
                return ExplainResponse(explanation=explanation, recommendation=recommendation)
        except Exception:
            pass  # fall through to rule-based

    # Rule-based fallback
    explanation = (
        f"Customer likely to {pred} based on {', '.join(factors[:2])}. "
        f"Model confidence: {prob*100:.1f}%."
    )
    recommendation = (
        "RM should discuss term deposit options and highlight current rates."
        if prob >= 0.5 else
        "Low likelihood — focus call on understanding objections before pitching."
    )
    return ExplainResponse(explanation=explanation, recommendation=recommendation)


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
