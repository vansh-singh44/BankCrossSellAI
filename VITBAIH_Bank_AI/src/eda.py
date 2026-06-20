"""
eda.py — Exploratory Data Analysis for Bank Marketing Dataset
Generates all required plots and answers business questions.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "bank-additional-full.csv"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

PALETTE = {"yes": "#2ecc71", "no": "#e74c3c"}
DARK_BG = "#1a1a2e"
ACCENT = "#16213e"


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, name: str) -> None:
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path.name}")


def styled_fig(nrows: int = 1, ncols: int = 1, figsize: tuple = (10, 6)) -> tuple[plt.Figure, any]:
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, facecolor=DARK_BG)
    if hasattr(ax, "__iter__") and not isinstance(ax, np.ndarray):
        pass
    elif isinstance(ax, np.ndarray):
        for a in ax.flat:
            a.set_facecolor(ACCENT)
            a.tick_params(colors="white")
            a.spines[:].set_color("#444")
            a.yaxis.label.set_color("white")
            a.xaxis.label.set_color("white")
            a.title.set_color("white")
    else:
        ax.set_facecolor(ACCENT)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
    return fig, ax


# ─── Load ─────────────────────────────────────────────────────────────────────
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the bank marketing CSV (semicolon-delimited)."""
    df = pd.read_csv(path, sep=";")
    return df


# ─── Overview ─────────────────────────────────────────────────────────────────
def overview(df: pd.DataFrame) -> dict:
    """Print and return basic dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape          : {df.shape}")
    print(f"Columns        : {df.columns.tolist()}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nMissing values : {df.isnull().sum().sum()}")
    print(f"Duplicates     : {df.duplicated().sum()}")
    print(f"\nTarget distribution:\n{df['y'].value_counts()}")
    print(f"\nClass imbalance ratio: {df['y'].value_counts()['no'] / df['y'].value_counts()['yes']:.2f}:1")
    print(f"\nSummary statistics:\n{df.describe().T.to_string()}")

    return {
        "shape": df.shape,
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "yes_pct": round(df['y'].value_counts(normalize=True)['yes'] * 100, 2),
    }


# ─── Plot 1: Target Distribution ──────────────────────────────────────────────
def plot_target_distribution(df: pd.DataFrame) -> None:
    counts = df['y'].value_counts()
    pcts = df['y'].value_counts(normalize=True) * 100

    fig, ax = styled_fig(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[k] for k in counts.index],
                  edgecolor="white", linewidth=0.8)
    for bar, pct in zip(bars, pcts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 300,
                f"{bar.get_height():,}\n({pct:.1f}%)",
                ha="center", va="bottom", color="white", fontsize=12, fontweight="bold")
    ax.set_title("Target Distribution — Term Deposit Subscription", color="white", fontsize=14, pad=12)
    ax.set_xlabel("Subscribed (y)", color="white")
    ax.set_ylabel("Count", color="white")
    ax.set_ylim(0, counts.max() * 1.2)
    _save(fig, "01_target_distribution.png")


# ─── Plot 2: Age Distribution ─────────────────────────────────────────────────
def plot_age_distribution(df: pd.DataFrame) -> None:
    fig, ax = styled_fig(figsize=(12, 5))
    for label, grp in df.groupby("y"):
        ax.hist(grp["age"], bins=30, alpha=0.7, label=label,
                color=PALETTE[label], edgecolor="white", linewidth=0.4)
    ax.set_title("Age Distribution by Subscription Status", color="white", fontsize=14)
    ax.set_xlabel("Age", color="white")
    ax.set_ylabel("Frequency", color="white")
    leg = ax.legend(title="Subscribed", facecolor=ACCENT, labelcolor="white")
    leg.get_title().set_color("white")
    _save(fig, "02_age_distribution.png")


# ─── Plot 3: Balance Distribution (duration proxy) ───────────────────────────
def plot_duration_distribution(df: pd.DataFrame) -> None:
    """Use call duration as proxy for balance (dataset has no balance column)."""
    fig, ax = styled_fig(figsize=(12, 5))
    for label, grp in df.groupby("y"):
        data = grp["duration"].clip(0, 1500)
        ax.hist(data, bins=40, alpha=0.7, label=label,
                color=PALETTE[label], edgecolor="white", linewidth=0.4)
    ax.set_title("Call Duration Distribution by Subscription Status", color="white", fontsize=14)
    ax.set_xlabel("Duration (seconds)", color="white")
    ax.set_ylabel("Frequency", color="white")
    leg = ax.legend(title="Subscribed", facecolor=ACCENT, labelcolor="white")
    leg.get_title().set_color("white")
    _save(fig, "03_duration_distribution.png")


# ─── Plot 4: Job vs Subscription ─────────────────────────────────────────────
def plot_job_vs_subscription(df: pd.DataFrame) -> None:
    job_sub = (df.groupby("job")["y"]
               .value_counts(normalize=True)
               .unstack()
               .fillna(0)
               .sort_values("yes", ascending=True))

    fig, ax = styled_fig(figsize=(12, 7))
    job_sub[["no", "yes"]].plot(kind="barh", stacked=True, ax=ax,
                                 color=[PALETTE["no"], PALETTE["yes"]],
                                 edgecolor="white", linewidth=0.5)
    ax.set_title("Job Category vs Subscription Rate", color="white", fontsize=14)
    ax.set_xlabel("Proportion", color="white")
    ax.set_ylabel("Job", color="white")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    leg = ax.legend(["No", "Yes"], title="Subscribed", facecolor=ACCENT, labelcolor="white")
    leg.get_title().set_color("white")
    _save(fig, "04_job_vs_subscription.png")


# ─── Plot 5: Housing Loan vs Subscription ────────────────────────────────────
def plot_housing_vs_subscription(df: pd.DataFrame) -> None:
    ct = pd.crosstab(df["housing"], df["y"], normalize="index")
    fig, ax = styled_fig(figsize=(8, 5))
    ct.plot(kind="bar", ax=ax, color=[PALETTE["no"], PALETTE["yes"]],
            edgecolor="white", linewidth=0.8, rot=0)
    ax.set_title("Housing Loan vs Subscription Rate", color="white", fontsize=14)
    ax.set_xlabel("Has Housing Loan", color="white")
    ax.set_ylabel("Proportion", color="white")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    leg = ax.legend(["No", "Yes"], title="Subscribed", facecolor=ACCENT, labelcolor="white")
    leg.get_title().set_color("white")
    _save(fig, "05_housing_vs_subscription.png")


# ─── Plot 6: Age Group vs Subscription ───────────────────────────────────────
def plot_age_group_vs_subscription(df: pd.DataFrame) -> None:
    bins = [0, 25, 35, 45, 55, 65, 100]
    labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
    df2 = df.copy()
    df2["age_group"] = pd.cut(df2["age"], bins=bins, labels=labels, right=False)
    ct = pd.crosstab(df2["age_group"], df2["y"], normalize="index")

    fig, ax = styled_fig(figsize=(11, 5))
    ct.plot(kind="bar", ax=ax, color=[PALETTE["no"], PALETTE["yes"]],
            edgecolor="white", linewidth=0.8, rot=0)
    ax.set_title("Age Group vs Subscription Rate", color="white", fontsize=14)
    ax.set_xlabel("Age Group", color="white")
    ax.set_ylabel("Proportion", color="white")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    leg = ax.legend(["No", "Yes"], title="Subscribed", facecolor=ACCENT, labelcolor="white")
    leg.get_title().set_color("white")
    _save(fig, "06_age_group_vs_subscription.png")


# ─── Plot 7: Correlation Heatmap ─────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    df2 = df.copy()
    df2["y_bin"] = (df2["y"] == "yes").astype(int)
    corr = df2[num_cols + ["y_bin"]].corr()

    fig, ax = plt.subplots(figsize=(13, 10), facecolor=DARK_BG)
    ax.set_facecolor(ACCENT)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
                ax=ax, linewidths=0.5, linecolor="#333",
                annot_kws={"size": 8, "color": "white"},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", color="white", fontsize=14, pad=12)
    ax.tick_params(colors="white", labelsize=9)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    _save(fig, "07_correlation_heatmap.png")


# ─── Business Questions ───────────────────────────────────────────────────────
def answer_business_questions(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("BUSINESS INSIGHTS")
    print("=" * 60)

    # Q1: Which jobs subscribe most?
    job_yes = (df[df["y"] == "yes"]["job"].value_counts(normalize=True) * 100).round(2)
    job_rate = (df.groupby("job")["y"].apply(lambda x: (x == "yes").mean() * 100)).sort_values(ascending=False)
    print(f"\n[Q1] Highest subscription RATE by job:\n{job_rate.round(2).to_string()}")
    top_job = job_rate.idxmax()

    # Q2: Does duration (balance proxy) impact subscription?
    dur_yes = df[df["y"] == "yes"]["duration"].median()
    dur_no  = df[df["y"] == "no"]["duration"].median()
    print(f"\n[Q2] Median duration — Yes: {dur_yes:.0f}s | No: {dur_no:.0f}s")
    print(f"     Longer calls strongly correlate with subscription (ratio: {dur_yes/dur_no:.2f}x)")

    # Q3: Age-group behaviour
    bins = [0, 25, 35, 45, 55, 65, 100]
    labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
    df2 = df.copy()
    df2["age_group"] = pd.cut(df2["age"], bins=bins, labels=labels, right=False)
    age_rate = df2.groupby("age_group")["y"].apply(lambda x: (x == "yes").mean() * 100).round(2)
    print(f"\n[Q3] Subscription rate by age group:\n{age_rate.to_string()}")

    # Q4: Housing loan effect
    hl_rate = df.groupby("housing")["y"].apply(lambda x: (x == "yes").mean() * 100).round(2)
    print(f"\n[Q4] Subscription rate by housing loan:\n{hl_rate.to_string()}")

    return {
        "top_job_subscription_rate": top_job,
        "median_duration_yes": dur_yes,
        "median_duration_no": dur_no,
        "age_group_rates": age_rate.to_dict(),
        "housing_loan_rates": hl_rate.to_dict(),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def run_eda() -> tuple[pd.DataFrame, dict]:
    print("Loading data…")
    df = load_data()

    stats = overview(df)

    print("\nGenerating plots…")
    plot_target_distribution(df)
    plot_age_distribution(df)
    plot_duration_distribution(df)
    plot_job_vs_subscription(df)
    plot_housing_vs_subscription(df)
    plot_age_group_vs_subscription(df)
    plot_correlation_heatmap(df)

    insights = answer_business_questions(df)
    stats.update(insights)

    print("\n✓ Phase 1 EDA complete. Plots saved to plots/")
    return df, stats


if __name__ == "__main__":
    run_eda()
