"""Pairwise hyperparameter interaction detection.

Measures how much the effect of one hyperparameter depends on the
value of another. Strong interactions indicate that parameters
cannot be tuned independently.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from exatune.analysis._utils import infer_hyperparameter_columns, validate_and_filter


def compute_pairwise_interactions(
    df: pd.DataFrame,
    metric: str = "mean_score",
    param_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute pairwise interaction strengths between all parameter pairs.

    Interaction strength is defined as the additional variance explained
    by the pair beyond the sum of their individual (main effect) variances.
    This is analogous to the interaction term in a two-way ANOVA.

    Args:
        df: Results DataFrame.
        metric: Metric column.
        param_columns: Parameters to analyze.

    Returns:
        DataFrame with columns: param_1, param_2, main_effect_1,
        main_effect_2, joint_effect, interaction_strength.
    """
    filtered, metric_col, param_columns = validate_and_filter(df, metric, param_columns)

    if len(param_columns) < 2:
        return pd.DataFrame(
            columns=[
                "param_1",
                "param_2",
                "main_effect_1",
                "main_effect_2",
                "joint_effect",
                "interaction_strength",
            ]
        )

    n = len(filtered)
    grand_mean = filtered[metric_col].mean()
    ss_total = filtered[metric_col].var() * (n - 1)

    if ss_total == 0:
        rows = []
        for i in range(len(param_columns)):
            for j in range(i + 1, len(param_columns)):
                rows.append(
                    {
                        "param_1": param_columns[i],
                        "param_2": param_columns[j],
                        "main_effect_1": 0.0,
                        "main_effect_2": 0.0,
                        "joint_effect": 0.0,
                        "interaction_strength": 0.0,
                    }
                )
        return pd.DataFrame(rows)

    # Compute individual main effects (eta-squared)
    main_effects = {}
    for param in param_columns:
        groups = filtered.groupby(param)[metric_col]
        group_means = groups.mean()
        group_sizes = groups.size()
        ss_between = sum(
            size * (mean - grand_mean) ** 2
            for mean, size in zip(group_means.values, group_sizes.values)
        )
        main_effects[param] = ss_between / ss_total

    # Compute pairwise joint effects
    rows = []
    for i in range(len(param_columns)):
        for j in range(i + 1, len(param_columns)):
            p1, p2 = param_columns[i], param_columns[j]

            groups = filtered.groupby([p1, p2])[metric_col]
            group_means = groups.mean()
            group_sizes = groups.size()
            ss_joint = sum(
                size * (mean - grand_mean) ** 2
                for mean, size in zip(group_means.values, group_sizes.values)
            )
            joint_effect = ss_joint / ss_total

            # Interaction = joint - main1 - main2
            interaction = max(0.0, joint_effect - main_effects[p1] - main_effects[p2])

            rows.append(
                {
                    "param_1": p1,
                    "param_2": p2,
                    "main_effect_1": float(main_effects[p1]),
                    "main_effect_2": float(main_effects[p2]),
                    "joint_effect": float(joint_effect),
                    "interaction_strength": float(interaction),
                }
            )

    result = pd.DataFrame(rows)
    return result.sort_values("interaction_strength", ascending=False).reset_index(drop=True)


def detect_significant_interactions(
    df: pd.DataFrame,
    metric: str = "mean_score",
    param_columns: Optional[List[str]] = None,
    threshold: float = 0.01,
) -> List[Tuple[str, str, float]]:
    """Identify parameter pairs with significant interaction effects.

    Args:
        df: Results DataFrame.
        metric: Metric column.
        param_columns: Parameters to analyze.
        threshold: Minimum interaction strength to consider significant.

    Returns:
        List of (param1, param2, interaction_strength) tuples,
        sorted by strength descending.
    """
    interactions = compute_pairwise_interactions(df, metric, param_columns)

    significant = interactions[interactions["interaction_strength"] >= threshold]

    return [
        (row["param_1"], row["param_2"], row["interaction_strength"])
        for _, row in significant.iterrows()
    ]
