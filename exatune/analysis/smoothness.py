"""Landscape smoothness metrics for hyperparameter performance surfaces.

Quantifies how smooth or rugged the objective landscape is, which
indicates whether local optimization methods would be effective
on this hyperparameter space.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from exatune.analysis._utils import (
    build_adjacency_structure,
    infer_hyperparameter_columns,
    validate_and_filter,
)


def compute_autocorrelation(
    df: pd.DataFrame,
    metric: str = "mean_score",
    param_columns: Optional[List[str]] = None,
    max_lag: int = 3,
) -> Dict[str, Any]:
    """Compute spatial autocorrelation of the metric on the grid.

    Uses Moran's I statistic adapted for the hyperparameter grid.
    High autocorrelation (close to 1) means the landscape is smooth;
    values near 0 indicate randomness; negative values indicate
    checkerboard-like patterns.

    Args:
        df: Results DataFrame.
        metric: Metric column name.
        param_columns: Hyperparameter columns (None = auto-detect).
        max_lag: Maximum neighborhood distance to compute.

    Returns:
        Dict with keys:
            'morans_i': overall Moran's I statistic
            'expected_i': expected I under null hypothesis
            'lag_correlations': autocorrelation at each lag distance
            'interpretation': human-readable interpretation
    """
    filtered, metric_col, param_columns = validate_and_filter(df, metric, param_columns)
    filtered = filtered.reset_index(drop=True)

    values = filtered[metric_col].values.astype(float)
    n = len(values)
    mean_val = values.mean()
    deviations = values - mean_val
    ss = np.sum(deviations ** 2)

    if ss == 0 or n < 3:
        return {
            "morans_i": 0.0,
            "expected_i": -1.0 / (n - 1) if n > 1 else 0.0,
            "lag_correlations": [0.0] * max_lag,
            "interpretation": "All values identical; smoothness undefined.",
        }

    adjacency = build_adjacency_structure(filtered, param_columns)

    # Compute Moran's I for lag-1 neighbors
    numerator = 0.0
    w_sum = 0.0
    for i in range(n):
        for j in adjacency[i]:
            numerator += deviations[i] * deviations[j]
            w_sum += 1.0

    if w_sum == 0:
        morans_i = 0.0
    else:
        morans_i = float((n / w_sum) * (numerator / ss))

    expected_i = -1.0 / (n - 1)

    # Compute autocorrelation at multiple lag distances
    lag_correlations = []
    for lag in range(1, max_lag + 1):
        lag_neighbors = _get_lag_neighbors(adjacency, lag)
        lag_num = 0.0
        lag_w = 0.0
        for i in range(n):
            for j in lag_neighbors.get(i, []):
                lag_num += deviations[i] * deviations[j]
                lag_w += 1.0
        if lag_w > 0 and ss > 0:
            lag_correlations.append(float((n / lag_w) * (lag_num / ss)))
        else:
            lag_correlations.append(0.0)

    # Interpret
    if morans_i > 0.5:
        interpretation = "Highly smooth landscape. Local search methods should work well."
    elif morans_i > 0.2:
        interpretation = "Moderately smooth landscape. Local search may find good solutions."
    elif morans_i > -0.1:
        interpretation = "Weakly structured landscape. Random or exhaustive search recommended."
    else:
        interpretation = "Anti-correlated (checkerboard) landscape. Exhaustive search essential."

    return {
        "morans_i": morans_i,
        "expected_i": expected_i,
        "lag_correlations": lag_correlations,
        "interpretation": interpretation,
    }


def _get_lag_neighbors(
    adjacency: Dict[int, List[int]], lag: int
) -> Dict[int, List[int]]:
    """Get neighbors at exactly the given lag distance via BFS."""
    if lag == 1:
        return adjacency

    result = {}
    for start in adjacency:
        visited = {start}
        current_layer = {start}
        for _ in range(lag):
            next_layer = set()
            for node in current_layer:
                for neighbor in adjacency.get(node, []):
                    if neighbor not in visited:
                        next_layer.add(neighbor)
                        visited.add(neighbor)
            current_layer = next_layer
        result[start] = list(current_layer)

    return result


def compute_gradient_statistics(
    df: pd.DataFrame,
    metric: str = "mean_score",
    param_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute gradient-based smoothness statistics.

    Computes finite differences between neighboring configurations
    and summarizes the distribution of gradient magnitudes.

    Args:
        df: Results DataFrame.
        metric: Metric column name.
        param_columns: Hyperparameter columns.

    Returns:
        Dict with keys:
            'mean_gradient': mean absolute gradient
            'max_gradient': maximum gradient (steepest cliff)
            'std_gradient': std of gradients
            'gradient_cv': coefficient of variation
            'roughness': ratio of gradient std to metric range
            'per_param_gradients': mean gradient per parameter direction
    """
    filtered, metric_col, param_columns = validate_and_filter(df, metric, param_columns)
    filtered = filtered.reset_index(drop=True)

    values = filtered[metric_col].values.astype(float)
    adjacency = build_adjacency_structure(filtered, param_columns)

    # Collect all finite differences
    gradients = []
    seen_pairs = set()
    for i in range(len(filtered)):
        for j in adjacency[i]:
            pair = (min(i, j), max(i, j))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                gradients.append(abs(values[i] - values[j]))

    if not gradients:
        return {
            "mean_gradient": 0.0,
            "max_gradient": 0.0,
            "std_gradient": 0.0,
            "gradient_cv": 0.0,
            "roughness": 0.0,
            "per_param_gradients": {p: 0.0 for p in param_columns},
        }

    gradients = np.array(gradients)
    metric_range = values.max() - values.min()

    # Per-parameter gradients
    per_param = {}
    for p_idx, param in enumerate(param_columns):
        param_grads = []
        for i in range(len(filtered)):
            for j in adjacency[i]:
                # Check if this neighbor differs only in this parameter
                differs_in = []
                for k, p in enumerate(param_columns):
                    if filtered[p].iloc[i] != filtered[p].iloc[j]:
                        differs_in.append(k)
                if differs_in == [p_idx]:
                    param_grads.append(abs(values[i] - values[j]))
        per_param[param] = float(np.mean(param_grads)) if param_grads else 0.0

    mean_grad = float(np.mean(gradients))
    std_grad = float(np.std(gradients))

    return {
        "mean_gradient": mean_grad,
        "max_gradient": float(np.max(gradients)),
        "std_gradient": std_grad,
        "gradient_cv": float(std_grad / mean_grad) if mean_grad > 0 else 0.0,
        "roughness": float(std_grad / metric_range) if metric_range > 0 else 0.0,
        "per_param_gradients": per_param,
    }


def compute_ruggedness(
    df: pd.DataFrame,
    metric: str = "mean_score",
    param_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute the ruggedness index of the landscape.

    Ruggedness measures how much the landscape deviates from a
    smooth trend. Computed as the ratio of local variance
    (variance among neighbors) to global variance.

    Args:
        df: Results DataFrame.
        metric: Metric column name.
        param_columns: Hyperparameter columns.

    Returns:
        Dict with keys:
            'ruggedness_index': 0 (smooth) to 1 (maximally rugged)
            'local_variance': average variance in neighborhoods
            'global_variance': variance across all configurations
            'interpretation': human-readable interpretation
    """
    filtered, metric_col, param_columns = validate_and_filter(df, metric, param_columns)
    filtered = filtered.reset_index(drop=True)

    values = filtered[metric_col].values.astype(float)
    global_var = float(np.var(values))

    if global_var == 0:
        return {
            "ruggedness_index": 0.0,
            "local_variance": 0.0,
            "global_variance": 0.0,
            "interpretation": "All values identical; landscape is trivially smooth.",
        }

    adjacency = build_adjacency_structure(filtered, param_columns)

    # Compute local variance: for each node, variance among it and its neighbors
    local_vars = []
    for i in range(len(filtered)):
        neighborhood = [values[i]] + [values[j] for j in adjacency[i]]
        if len(neighborhood) > 1:
            local_vars.append(np.var(neighborhood))

    local_var = float(np.mean(local_vars)) if local_vars else 0.0

    # Ruggedness = local_var / global_var
    # High ratio means local neighborhoods are nearly as variable as the whole
    ruggedness = min(1.0, local_var / global_var) if global_var > 0 else 0.0

    if ruggedness < 0.2:
        interpretation = "Smooth landscape. Performance changes gradually."
    elif ruggedness < 0.5:
        interpretation = "Moderately rugged. Some abrupt performance changes."
    elif ruggedness < 0.8:
        interpretation = "Rugged landscape. Performance varies substantially between neighbors."
    else:
        interpretation = "Highly rugged. Near-random variation; exhaustive search essential."

    return {
        "ruggedness_index": float(ruggedness),
        "local_variance": local_var,
        "global_variance": global_var,
        "interpretation": interpretation,
    }
