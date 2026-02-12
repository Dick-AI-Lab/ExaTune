"""Multimodality detection: counting and characterizing local optima.

Provides tools for identifying local optima in the hyperparameter landscape,
computing basin of attraction sizes, and measuring fitness-distance
correlation to assess landscape difficulty.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from exatune.analysis._utils import (
    build_adjacency_structure,
    compute_grid_distance,
    get_param_value_orders,
    validate_and_filter,
)


def find_local_optima(
    df: pd.DataFrame,
    metric: str = "mean_score",
    direction: str = "maximize",
    param_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Find all local optima in the hyperparameter landscape.

    A configuration is a local optimum if its metric value is
    better than or equal to all its grid neighbors.

    Args:
        df: Results DataFrame.
        metric: Metric to analyze.
        direction: 'maximize' or 'minimize'.
        param_columns: Hyperparameter columns.

    Returns:
        DataFrame containing only local optima rows, with additional
        columns 'optimality_gap' (difference from global optimum).
    """
    filtered, metric_col, param_columns = validate_and_filter(df, metric, param_columns)
    filtered = filtered.reset_index(drop=True)

    values = filtered[metric_col].values.astype(float)
    adjacency = build_adjacency_structure(filtered, param_columns)

    is_better = (lambda a, b: a >= b) if direction == "maximize" else (lambda a, b: a <= b)

    optima_indices = []
    for i in range(len(filtered)):
        neighbors = adjacency[i]
        if not neighbors:
            # Isolated point is trivially a local optimum
            optima_indices.append(i)
            continue
        if all(is_better(values[i], values[j]) for j in neighbors):
            optima_indices.append(i)

    optima_df = filtered.iloc[optima_indices].copy()

    # Compute optimality gap
    if direction == "maximize":
        global_best = values.max()
        optima_df["optimality_gap"] = global_best - optima_df[metric_col].values
    else:
        global_best = values.min()
        optima_df["optimality_gap"] = optima_df[metric_col].values - global_best

    return optima_df.reset_index(drop=True)


def compute_multimodality_metrics(
    df: pd.DataFrame,
    metric: str = "mean_score",
    direction: str = "maximize",
    param_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute metrics characterizing the multimodality of the landscape.

    Args:
        df: Results DataFrame.
        metric: Metric to analyze.
        direction: 'maximize' or 'minimize'.
        param_columns: Hyperparameter columns.

    Returns:
        Dict with keys describing multimodality characteristics.
    """
    filtered, metric_col, param_columns = validate_and_filter(df, metric, param_columns)
    filtered = filtered.reset_index(drop=True)

    optima = find_local_optima(filtered, metric_col, direction, param_columns)
    n_optima = len(optima)
    n_configs = len(filtered)

    values = filtered[metric_col].values.astype(float)
    optima_values = optima[metric_col].values.astype(float)

    if direction == "maximize":
        global_optimum = float(values.max())
        sorted_optima = np.sort(optima_values)[::-1]
    else:
        global_optimum = float(values.min())
        sorted_optima = np.sort(optima_values)

    second_best = float(sorted_optima[1]) if len(sorted_optima) > 1 else global_optimum

    # Funnel index: fraction of configurations within 5% of global optimum
    metric_range = values.max() - values.min()
    if metric_range > 0:
        threshold = 0.05 * metric_range
        if direction == "maximize":
            near_optimal = np.sum(values >= global_optimum - threshold)
        else:
            near_optimal = np.sum(values <= global_optimum + threshold)
        funnel_index = float(near_optimal / n_configs)
    else:
        funnel_index = 1.0

    if n_optima == 1:
        interpretation = "Unimodal landscape. Single clear optimum; local search ideal."
    elif n_optima <= 3:
        interpretation = "Low multimodality. Few local optima; local search likely effective."
    elif n_optima <= n_configs * 0.1:
        interpretation = "Moderate multimodality. Multiple local optima present."
    else:
        interpretation = "Highly multimodal. Many local optima; exhaustive search justified."

    return {
        "n_local_optima": n_optima,
        "n_configurations": n_configs,
        "optima_density": float(n_optima / n_configs) if n_configs > 0 else 0.0,
        "global_optimum_value": global_optimum,
        "second_best_optimum_value": second_best,
        "gap_global_to_second": abs(global_optimum - second_best),
        "mean_optimum_value": float(np.mean(optima_values)),
        "std_optimum_values": float(np.std(optima_values)) if len(optima_values) > 1 else 0.0,
        "funnel_index": funnel_index,
        "interpretation": interpretation,
    }


def compute_fitness_distance_correlation(
    df: pd.DataFrame,
    metric: str = "mean_score",
    direction: str = "maximize",
    param_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute fitness-distance correlation (FDC).

    FDC measures the correlation between a configuration's metric value
    and its distance from the global optimum. High positive FDC indicates
    a "big valley" structure where getting closer to the optimum reliably
    improves performance. Low FDC indicates a deceptive landscape.

    Args:
        df: Results DataFrame.
        metric: Metric to analyze.
        direction: 'maximize' or 'minimize'.
        param_columns: Hyperparameter columns.

    Returns:
        Dict with keys:
            'fdc': fitness-distance correlation (-1 to 1)
            'interpretation': difficulty assessment
    """
    filtered, metric_col, param_columns = validate_and_filter(df, metric, param_columns)
    filtered = filtered.reset_index(drop=True)

    values = filtered[metric_col].values.astype(float)

    # Find global optimum index
    if direction == "maximize":
        best_idx = int(np.argmax(values))
    else:
        best_idx = int(np.argmin(values))

    value_orders = get_param_value_orders(filtered, param_columns)

    # Compute distance from each config to the global optimum
    distances = []
    for i in range(len(filtered)):
        d = compute_grid_distance(filtered, i, best_idx, param_columns, value_orders)
        distances.append(d)

    distances = np.array(distances, dtype=float)

    # For maximize: we expect negative correlation (closer = higher score)
    # For minimize: we expect positive correlation (closer = lower score)
    if direction == "maximize":
        fitness = -values  # Negate so higher fitness = lower value for correlation
    else:
        fitness = values

    # Compute Pearson correlation between fitness and distance
    if np.std(distances) == 0 or np.std(fitness) == 0:
        fdc = 0.0
    else:
        fdc = float(np.corrcoef(fitness, distances)[0, 1])

    # Interpret: high positive FDC means fitness correlates with distance
    # (i.e., closer to optimum = better fitness) = easy landscape
    if fdc > 0.15:
        interpretation = (
            "Strong fitness-distance correlation ('big valley'). "
            "Local search should find near-optimal solutions."
        )
    elif fdc > -0.15:
        interpretation = (
            "Weak fitness-distance correlation. "
            "Landscape has moderate structure; results may vary by starting point."
        )
    else:
        interpretation = (
            "Negative fitness-distance correlation (deceptive landscape). "
            "Moving toward the optimum may not improve fitness. Exhaustive search essential."
        )

    return {
        "fdc": fdc,
        "interpretation": interpretation,
    }
