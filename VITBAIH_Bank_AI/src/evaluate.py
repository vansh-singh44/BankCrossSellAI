"""
evaluate.py — Model evaluation: metrics, confusion matrix, classification report,
              model comparison, and feature importance plot.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

DARK_BG = "#1a1a2e"
ACCENT  = "#16213e"


def _save(fig: plt.Figure, name: str) -> None:
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Plot saved → {path.name}")


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
) -> dict[str, float]:
    """Compute and print all evaluation metrics."""
    metrics = {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
    }
    print(f"\n{'─'*50}")
    print(f"  {model_name}")
    print(f"{'─'*50}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:<12}: {v}")
    print(f"\nClassification Report:\n{classification_report(y_true, y_pred, target_names=['No','Yes'])}")
    return metrics


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    filename: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=DARK_BG)
    ax.set_facecolor(ACCENT)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No", "Yes"], yticklabels=["No", "Yes"],
        ax=ax, linewidths=0.5, linecolor="#333",
        annot_kws={"size": 14, "color": "white"},
    )
    ax.set_title(f"Confusion Matrix — {model_name}", color="white", fontsize=12, pad=10)
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("Actual", color="white")
    ax.tick_params(colors="white")
    _save(fig, filename)


def plot_roc_curves(
    roc_data: list[dict],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=DARK_BG)
    ax.set_facecolor(ACCENT)
    colors = ["#2ecc71", "#3498db", "#e67e22"]
    for i, d in enumerate(roc_data):
        fpr, tpr, _ = roc_curve(d["y_true"], d["y_prob"])
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f"{d['name']} (AUC={d['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate", color="white")
    ax.set_title("ROC Curves — Model Comparison", color="white", fontsize=13)
    ax.tick_params(colors="white")
    leg = ax.legend(facecolor=ACCENT, labelcolor="white")
    _save(fig, "08_roc_curves.png")


def plot_feature_importance(model: CatBoostClassifier, feature_names: list[str]) -> None:
    importances = model.get_feature_importance()
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    top = fi.tail(15)

    fig, ax = plt.subplots(figsize=(10, 7), facecolor=DARK_BG)
    ax.set_facecolor(ACCENT)
    bars = ax.barh(top.index, top.values, color="#3498db", edgecolor="white", linewidth=0.5)
    ax.set_title("Top 15 Feature Importances — CatBoost", color="white", fontsize=13)
    ax.set_xlabel("Importance Score", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    _save(fig, "09_feature_importance.png")


def plot_model_comparison(results: list[dict]) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    df = pd.DataFrame(results).set_index("model")[metrics]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    ax.set_facecolor(ACCENT)
    x = np.arange(len(metrics))
    width = 0.35
    colors = ["#e74c3c", "#2ecc71"]
    for i, (idx, row) in enumerate(df.iterrows()):
        ax.bar(x + i * width, row.values, width, label=idx, color=colors[i],
               edgecolor="white", linewidth=0.6)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([m.upper() for m in metrics], color="white")
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Comparison — LR vs CatBoost", color="white", fontsize=13)
    ax.set_ylabel("Score", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    leg = ax.legend(facecolor=ACCENT, labelcolor="white")
    _save(fig, "10_model_comparison.png")


def evaluate_models(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    lr_model: Any,
    cat_model: CatBoostClassifier,
    cat_features: list[str],
) -> list[dict]:
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # LR
    lr_pred = lr_model.predict(X_test)
    lr_prob = lr_model.predict_proba(X_test)[:, 1]
    lr_metrics = compute_metrics(y_test, lr_pred, lr_prob, "Logistic Regression")
    plot_confusion_matrix(y_test, lr_pred, "Logistic Regression", "08a_cm_lr.png")

    # CatBoost
    cat_idx = [X_test.columns.tolist().index(c) for c in cat_features if c in X_test.columns]
    pool    = Pool(X_test, cat_features=cat_idx)
    cat_pred = cat_model.predict(pool).flatten()
    cat_prob = cat_model.predict_proba(pool)[:, 1]
    cat_metrics = compute_metrics(y_test, cat_pred, cat_prob, "CatBoost")
    plot_confusion_matrix(y_test, cat_pred, "CatBoost", "08b_cm_catboost.png")

    results = [lr_metrics, cat_metrics]

    plot_roc_curves([
        {"name": "Logistic Regression", "y_true": y_test, "y_prob": lr_prob, "auc": lr_metrics["roc_auc"]},
        {"name": "CatBoost",            "y_true": y_test, "y_prob": cat_prob, "auc": cat_metrics["roc_auc"]},
    ])

    plot_feature_importance(cat_model, X_test.columns.tolist())
    plot_model_comparison(results)

    print("\n📊 Observations:")
    print(f"  • CatBoost ROC-AUC ({cat_metrics['roc_auc']}) vs LR ({lr_metrics['roc_auc']})")
    print(f"  • CatBoost F1 ({cat_metrics['f1']}) vs LR ({lr_metrics['f1']})")
    print("  • CatBoost better handles non-linearity and native categoricals.")
    print("  • F1 is the primary metric due to class imbalance.")

    print("\n✓ Evaluation complete.")
    return results


if __name__ == "__main__":
    import joblib
    from preprocess import CAT_FEATURES, run_preprocessing

    X_train, X_test, y_train, y_test = run_preprocessing()
    lr_model  = joblib.load(ROOT / "models" / "lr_model.pkl")
    cat_model = joblib.load(ROOT / "models" / "model.pkl")
    evaluate_models(X_test, y_test, lr_model, cat_model, CAT_FEATURES)
