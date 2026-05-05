"""Shared utilities for landscape analysis.

Provides grid adjacency structure, numeric encoding for categorical
parameters, and neighbor-finding functions that underpin smoothness,
multimodality, and gradient-based analyses.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Known metadata columns (same as visualization)
_METADATA_COLUMNS = {
    "job_id", "config_hash", "timestamp", "success",
    "mean_score", "std_score", "mean_train_score", "std_train_score",
    "fit_time_mean", "fit_time_std", "score_time_mean", "score_time_std",
    "rank", "best_score", "error_message", "error_type",
    # co2 tracking columns
    "emissions_kg_co2", "energy_consumed_kwh",
    "kg_co2", "energy_kwh",
    # row index
    "Unnamed: 0",
}

_METRIC_SUFFIXES = ("_mean", "_std", "_min", "_max")


def infer_hyperparameter_columns(df: pd.DataFrame) -> List[str]:
    """Infer which columns are hyperparameters vs metadata.

    Args:
        df: Results DataFrame.

    Returns:
        List of hyperparameter column names.
    """
    hp_cols = []
    for col in df.columns:
        if col in _METADATA_COLUMNS:
            continue
        if any(col.endswith(suffix) for suffix in _METRIC_SUFFIXES):
            continue
        hp_cols.append(col)
    return hp_cols


def get_metric_column(df: pd.DataFrame, metric: str) -> str:
    """Resolve a metric name to its DataFrame column name.

    Args:
        df: Results DataFrame.
        metric: User-specified metric name.

    Returns:
        Actual column name in the DataFrame.

    Raises:
        ValueError: If metric column cannot be resolved.
    """
    if metric in df.columns:
        return metric

    suffixed = f"{metric}_mean"
    if suffixed in df.columns:
        return suffixed

    aliases = {
        "accuracy": "mean_score",
        "score": "mean_score",
        "train_score": "mean_train_score",
    }
    if metric in aliases and aliases[metric] in df.columns:
        return aliases[metric]

    raise ValueError(
        f"Cannot resolve metric '{metric}' to a column. "
        f"Available columns: {sorted(df.columns.tolist())}"
    )


def validate_and_filter(
    df: pd.DataFrame,
    metric: str = "mean_score",
    param_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, str, List[str]]:
    """Validate DataFrame and return filtered data with resolved names.

    Args:
        df: Results DataFrame.
        metric: Metric name to resolve.
        param_columns: Hyperparameter columns (None = auto-detect).

    Returns:
        Tuple of (filtered_df, metric_col, param_columns).

    Raises:
        ValueError: If DataFrame is empty or invalid.
    """
    if df.empty:
        raise ValueError("Results DataFrame is empty.")

    metric_col = get_metric_column(df, metric)

    filtered = df.copy()
    if "success" in filtered.columns:
        filtered = filtered[filtered["success"] == True]  # noqa: E712

    filtered = filtered.dropna(subset=[metric_col])

    if filtered.empty:
        raise ValueError("No valid results after filtering.")

    if param_columns is None:
        param_columns = infer_hyperparameter_columns(filtered)

    if not param_columns:
        raise ValueError("No hyperparameter columns found.")

    return filtered, metric_col, param_columns


def encode_parameters_numeric(
    df: pd.DataFrame,
    param_columns: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[Any, int]]]:
    """Encode categorical hyperparameters as numeric values.

    For each parameter, maps unique values to consecutive integers
    preserving the natural order (numeric) or alphabetical order (string).

    Args:
        df: Results DataFrame.
        param_columns: Hyperparameter column names.

    Returns:
        Tuple of (encoded DataFrame copy, mapping dict per parameter).
    """
    encoded = df.copy()
    mappings = {}

    for param in param_columns:
        unique_vals = sorted(
            df[param].dropna().unique(),
            key=lambda x: (isinstance(x, str), x),
        )
        mapping = {v: i for i, v in enumerate(unique_vals)}
        encoded[param] = df[param].map(mapping)
        mappings[param] = mapping

    return encoded, mappings


def get_param_value_orders(
    df: pd.DataFrame,
    param_columns: List[str],
) -> Dict[str, List[Any]]:
    """Get sorted unique values for each parameter.

    Args:
        df: Results DataFrame.
        param_columns: Hyperparameter column names.

    Returns:
        Dict mapping parameter name to sorted list of unique values.
    """
    orders = {}
    for param in param_columns:
        vals = df[param].dropna().unique()
        try:
            orders[param] = sorted(vals, key=lambda x: (isinstance(x, str), x))
        except TypeError:
            orders[param] = list(vals)
    return orders


def build_adjacency_structure(
    df: pd.DataFrame,
    param_columns: List[str],
) -> Dict[int, List[int]]:
    df = df.reset_index(drop=True)
    value_orders = get_param_value_orders(df, param_columns)

    val_to_pos = {}
    for param in param_columns:
        val_to_pos[param] = {v: i for i, v in enumerate(value_orders[param])}

    # FIX: store ALL row indices per config key, not just the last
    config_index: Dict[tuple, List[int]] = {}
    for idx in range(len(df)):
        key = tuple(df[param].iloc[idx] for param in param_columns)
        config_index.setdefault(key, []).append(idx)

    adjacency = {idx: [] for idx in range(len(df))}

    for idx in range(len(df)):
        current_vals = [df[param].iloc[idx] for param in param_columns]

        for p_idx, param in enumerate(param_columns):
            current_val = current_vals[p_idx]
            current_pos = val_to_pos[param].get(current_val)
            if current_pos is None:
                continue

            order = value_orders[param]

            for delta in [-1, 1]:
                neighbor_pos = current_pos + delta
                if 0 <= neighbor_pos < len(order):
                    neighbor_key = list(current_vals)
                    neighbor_key[p_idx] = order[neighbor_pos]
                    neighbor_key = tuple(neighbor_key)

                    # FIX: iterate over all rows with that config
                    for neighbor_idx in config_index.get(neighbor_key, []):
                        if neighbor_idx != idx and neighbor_idx not in adjacency[idx]:
                            adjacency[idx].append(neighbor_idx)

    return adjacency
def compute_grid_distance(
    df: pd.DataFrame,
    idx1: int,
    idx2: int,
    param_columns: List[str],
    value_orders: Optional[Dict[str, List[Any]]] = None,
) -> int:
    """Compute Manhattan distance between two configurations on the grid.

    Distance is the sum of step differences across all parameters.

    Args:
        df: Results DataFrame.
        idx1: First configuration index.
        idx2: Second configuration index.
        param_columns: Hyperparameter columns.
        value_orders: Pre-computed value orderings (optional).

    Returns:
        Integer Manhattan distance.
    """
    if value_orders is None:
        value_orders = get_param_value_orders(df, param_columns)

    distance = 0
    for param in param_columns:
        order = value_orders[param]
        val_to_pos = {v: i for i, v in enumerate(order)}
        pos1 = val_to_pos.get(df[param].iloc[idx1], 0)
        pos2 = val_to_pos.get(df[param].iloc[idx2], 0)
        distance += abs(pos1 - pos2)

    return distance
