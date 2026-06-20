"""
preprocess.py — Feature engineering and preprocessing for Bank Marketing dataset.

Uses CatBoost-native categorical handling where possible.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "bank-additional-full.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Categorical columns that CatBoost will handle natively
CAT_FEATURES = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month",
    "day_of_week", "poutcome",
]

# Features to drop (duration is dropped for real-world inference
# since it's only known after the call)
FEATURES_TO_DROP: list[str] = []  # keep all; note in EXPLANATION.md


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the semicolon-delimited CSV."""
    return pd.read_csv(path, sep=";")


def encode_target(df: pd.DataFrame, col: str = "y") -> pd.DataFrame:
    """Convert yes → 1, no → 0."""
    df = df.copy()
    df[col] = df[col].map({"yes": 1, "no": 0})
    return df


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split into X and y.

    Categorical columns are kept as str dtype so CatBoost can detect them.
    """
    df = encode_target(df)
    y = df["y"]
    X = df.drop(columns=["y"] + FEATURES_TO_DROP)

    # Ensure cat features are str
    for col in CAT_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype(str)

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train-test split."""
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def save_artifacts(
    feature_names: list[str],
    encoder: LabelEncoder | None = None,
) -> None:
    """Persist feature list and optional encoder."""
    joblib.dump(feature_names, MODELS_DIR / "features.pkl")
    if encoder is not None:
        joblib.dump(encoder, MODELS_DIR / "encoder.pkl")
    print(f"  Saved features.pkl ({len(feature_names)} features)")


def run_preprocessing() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """End-to-end preprocessing pipeline. Returns X_train, X_test, y_train, y_test."""
    print("\n[Preprocessing] Loading data…")
    df = load_raw()
    print(f"  Shape: {df.shape}")

    X, y = build_features(df)
    print(f"  Features: {X.shape[1]}  |  Target distribution:\n"
          f"  {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    save_artifacts(X.columns.tolist())
    print("✓ Preprocessing complete.")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_preprocessing()
