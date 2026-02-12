"""Hyperparameter importance analysis via variance decomposition and correlation.

Provides multiple methods for quantifying how much each hyperparameter
affects the performance metric, enabling researchers to identify which
parameters matter most for their model.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from exatune.analysis._utils import infer_hyperparameter_columns, validate_and_filter


def compute_variance_importance(
    df: pd.DataFrame,
    metric: str = "mean_score",
    param_columns: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute variance-based importance for each hyperparameter.

    For each parameter, computes the fraction of total metric variance
    that is explained by grouping on that parameter (eta-squared,
    analogous to one-way ANOVA).

    Args:
        df: Results DataFrame.
        metric: Metric column.
        param_columns: Parameters to analyze (None = auto-detect).

    Returns:
        Dict mapping parameter name to importance score (0-1),
        sorted by importance descending.
    """
    filtered, metric_col, param_columns = validate_and_filter(df, metric, param_columns)

    total_var = filtered[metric_col].var()
    if total_var == 0:
        return {p: 0.0 for p in param_columns}

    importance = {}
    n = len(filtered)

    for param in param_columns:
        groups = filtered.groupby(param)[metric_col]
        # Between-group sum of squares / total sum of squares (eta-squared)
        group_means = groups.mean()
        grand_mean = filtered[metric_col].mean()
        group_sizes = groups.size()
        ss_between = sum(
            size * (mean - grand_mean) ** 2
            for mean, size in zip(group_means.values, group_sizes.values)
        )
        ss_total = total_var * (n - 1)
        importance[param] = float(ss_between / ss_total) if ss_total > 0 else 0.0

    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def compute_correlation_importance(
    df: pd.DataFrame,
    metric: str = "mean_score",
    param_columns: Optional[List[str]] = None,
    method: str = "spearman",
) -> Dict[str, float]:
    """Compute correlation-based importance for each hyperparameter.

    Uses Spearman rank correlation by default, which handles
    non-linear monotonic relationships.

    Args:
        df: Results DataFrame.
        metric: Metric column.
        param_columns: Parameters to analyze.
        method: Correlation method ('spearman', 'pearson', 'kendall').

    Returns:
        Dict mapping parameter name to absolute correlation value,
        sorted by importance descending.
    """
    filtered, metric_col, param_columns = validate_and_filter(df, metric, param_columns)

    importance = {}
    for param in param_columns:
        try:
            series = pd.to_numeric(filtered[param], errors="coerce")
            valid = series.notna()
            if valid.sum() > 2:
                corr = series[valid].corr(filtered.loc[valid, metric_col], method=method)
                importance[param] = abs(float(corr)) if pd.notna(corr) else 0.0
            else:
                importance[param] = 0.0
        except (TypeError, ValueError):
            # Categorical parameter: use eta-squared fallback
            groups = filtered.groupby(param)[metric_col]
            group_means = groups.mean()
            importance[param] = float(group_means.std() / filtered[metric_col].std()) if filtered[metric_col].std() > 0 else 0.0

    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def compute_range_importance(
    df: pd.DataFrame,
    metric: str = "mean_score",
    param_columns: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute range-based importance (difference of means).

    For each parameter, computes the range of mean metric values
    across its levels, normalized by the total metric range.

    Args:
        df: Results DataFrame.
        metric: Metric column.
        param_columns: Parameters to analyze.

    Returns:
        Dict mapping parameter name to normalized range importance (0-1).
    """
    filtered, metric_col, param_columns = validate_and_filter(df, metric, param_columns)

    total_range = filtered[metric_col].max() - filtered[metric_col].min()
    if total_range == 0:
        return {p: 0.0 for p in param_columns}

    importance = {}
    for param in param_columns:
        group_means = filtered.groupby(param)[metric_col].mean()
        param_range = group_means.max() - group_means.min()
        importance[param] = float(param_range / total_range)

    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def compute_importance_summary(
    df: pd.DataFrame,
    metric: str = "mean_score",
    param_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute all importance measures and return as a summary DataFrame.

    Args:
        df: Results DataFrame.
        metric: Metric column.
        param_columns: Parameters to analyze.

    Returns:
        DataFrame with columns: parameter, variance_importance,
        correlation_importance, range_importance, overall_rank.
    """
    filtered, metric_col, param_columns = validate_and_filter(df, metric, param_columns)

    var_imp = compute_variance_importance(filtered, metric_col, param_columns)
    corr_imp = compute_correlation_importance(filtered, metric_col, param_columns)
    range_imp = compute_range_importance(filtered, metric_col, param_columns)

    rows = []
    for param in param_columns:
        rows.append(
            {
                "parameter": param,
                "variance_importance": var_imp.get(param, 0.0),
                "correlation_importance": corr_imp.get(param, 0.0),
                "range_importance": range_imp.get(param, 0.0),
            }
        )

    summary_df = pd.DataFrame(rows)

    # Compute overall rank as mean of individual ranks
    for col in ["variance_importance", "correlation_importance", "range_importance"]:
        summary_df[f"{col}_rank"] = summary_df[col].rank(ascending=False)

    rank_cols = [c for c in summary_df.columns if c.endswith("_rank")]
    summary_df["mean_rank"] = summary_df[rank_cols].mean(axis=1)
    summary_df["overall_rank"] = summary_df["mean_rank"].rank().astype(int)
    summary_df = summary_df.drop(columns=rank_cols + ["mean_rank"])
    summary_df = summary_df.sort_values("overall_rank").reset_index(drop=True)

    return summary_df
