"""
train.py — Train Logistic Regression (baseline) and CatBoost (main) models.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

from preprocess import CAT_FEATURES, run_preprocessing


def build_lr_pipeline(cat_features: list[str]) -> Pipeline:
    """
    Logistic Regression baseline.
    Encodes categoricals with OrdinalEncoder, scales numerics.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler

    num_features = None  # resolved at fit time inside pipeline

    class _ColumnSelector:
        pass

    # Use ColumnTransformer inside Pipeline
    ct = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_features),
        ],
        remainder="passthrough",
    )
    return Pipeline([
        ("ct", ct),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
            C=1.0,
        )),
    ])


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Fit and return the LR baseline pipeline."""
    print("\n[Train] Fitting Logistic Regression baseline…")
    pipe = build_lr_pipeline(CAT_FEATURES)
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, MODELS_DIR / "lr_model.pkl")
    print("  LR saved → models/lr_model.pkl")
    return pipe


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> CatBoostClassifier:
    """Fit and return the CatBoost main model."""
    print("\n[Train] Fitting CatBoostClassifier…")

    cat_idx = [X_train.columns.tolist().index(c) for c in CAT_FEATURES if c in X_train.columns]

    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    eval_pool  = Pool(X_test,  y_test,  cat_features=cat_idx)

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        early_stopping_rounds=30,
        verbose=100,
    )
    model.fit(train_pool, eval_set=eval_pool, use_best_model=True)

    joblib.dump(model, MODELS_DIR / "model.pkl")
    print("  CatBoost saved → models/model.pkl")
    return model


def run_training() -> Tuple:
    X_train, X_test, y_train, y_test = run_preprocessing()
    lr_model   = train_logistic_regression(X_train, y_train)
    cat_model  = train_catboost(X_train, y_train, X_test, y_test)
    print("\n✓ Training complete.")
    return X_train, X_test, y_train, y_test, lr_model, cat_model


if __name__ == "__main__":
    run_training()
