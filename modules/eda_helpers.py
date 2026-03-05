"""Utilities for exploratory data analysis in notebook checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


_ORDINAL_MAPS = {
    "device_before_sleep": {
        "never": 0,
        "rarely (1-2 times a week)": 1,
        "sometimes (3-4 times a week)": 2,
        "often (5-6 times a week)": 3,
        "every night": 4,
    },
    "physical_activity": {
        "never": 0,
        "rarely (1-2 times a week)": 1,
        "sometimes (3-4 times a week)": 2,
        "often (5-6 times a week)": 3,
        "every day": 4,
    },
    "difficulty_falling_asleep": {
        "never": 0,
        "rarely (1-2 times a week)": 1,
        "sometimes (3-4 times a week)": 2,
        "often (5-6 times a week)": 3,
        "every night": 4,
    },
    "sleep_hours": {
        "less than 4 hours": 0,
        "4-5 hours": 1,
        "6-7 hours": 2,
        "7-8 hours": 3,
        "more than 8 hours": 4,
    },
    "sleep_quality": {
        "very poor": 0,
        "poor": 1,
        "average": 2,
        "good": 3,
        "very good": 4,
        "very bad": 0,
        "fairly bad": 1,
        "fairly good": 2,
    },
}


def ordinal_codes(series: pd.Series, column_name: str | None = None) -> pd.Series:
    """Convert ordinal string categories to meaningful numeric codes."""
    if pd.api.types.is_numeric_dtype(series):
        return series

    normalized = series.astype(str).str.strip().str.lower()
    mapping = _ORDINAL_MAPS.get(column_name or "", {})
    mapped = normalized.map(mapping) if mapping else pd.Series(index=series.index, dtype="float64")

    # If a column has values outside known mapping, fall back to stable categorical codes.
    if mapping and mapped.notna().any():
        fallback_mask = mapped.isna() & series.notna()
        if fallback_mask.any():
            fallback_codes = pd.Categorical(series[fallback_mask]).codes
            mapped.loc[fallback_mask] = fallback_codes
        return mapped

    codes = pd.Series(pd.Categorical(series).codes, index=series.index).astype("float64")
    codes[codes < 0] = np.nan
    return codes


def read_csv_first(paths: list[str]) -> tuple[pd.DataFrame, str]:
    """Load and return the first existing CSV from a prioritized list of paths."""
    for raw_path in paths:
        csv_path = Path(raw_path)
        if csv_path.exists():
            return pd.read_csv(csv_path), str(csv_path)
    raise FileNotFoundError(f"None of these files exist: {paths}")


def plot_likert_counts(df: pd.DataFrame, cols: list[str], title: str):
    """Plot distributions for a list of variables and return the matplotlib Figure."""
    import matplotlib.pyplot as plt

    present_cols = [col for col in cols if col in df.columns]
    n_cols = len(present_cols)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, col in enumerate(present_cols):
        series = df[col].dropna()
        if pd.api.types.is_numeric_dtype(series):
            sns.histplot(
                series,
                bins=min(12, max(1, series.nunique())),
                kde=False,
                ax=axes[i],
                color="#4C78A8",
            )
        else:
            order = series.value_counts().index
            sns.countplot(y=series, order=order, ax=axes[i], color="#4C78A8")
        axes[i].set_title(col.replace("_", " ").title())
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")

    for j in range(n_cols, 4):
        axes[j].axis("off")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def spearman_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a Spearman correlation matrix for selected columns."""
    present_cols = [col for col in cols if col in df.columns]
    work = df[present_cols].copy()
    for col in work.columns:
        if not pd.api.types.is_numeric_dtype(work[col]):
            work[col] = ordinal_codes(work[col], col)
    return work.corr(method="spearman")


def annotated_regplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    ax,
    label: str = "",
    text_x: float = 0.03,
    text_y: float = 0.97,
    text_ha: str = "left",
) -> None:
    """Draw a regression plot and annotate with Spearman correlation."""
    if x not in df.columns or y not in df.columns:
        ax.set_axis_off()
        return

    plot_df = df.dropna(subset=[x, y]).copy()
    if plot_df.empty:
        ax.set_axis_off()
        return

    sns.regplot(
        data=plot_df,
        x=x,
        y=y,
        ax=ax,
        scatter_kws={"alpha": 0.6},
        line_kws={"linewidth": 2},
        label=label,
    )
    corr, pval = spearmanr(plot_df[x], plot_df[y], nan_policy="omit")
    stat_label = f"rho={corr:.2f}, p={pval:.3g}, n={len(plot_df)}"
    if label:
        stat_label = f"{label}: {stat_label}"
    ax.text(
        text_x,
        text_y,
        stat_label,
        transform=ax.transAxes,
        va="top",
        ha=text_ha,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
