"""Comprehensive experiment summary statistics.

Aggregates all analysis metrics into a single summary dict and
provides formatted text output for console or report use.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from exatune.analysis._utils import infer_hyperparameter_columns, validate_and_filter
from exatune.analysis.importance import compute_importance_summary
from exatune.analysis.interactions import detect_significant_interactions
from exatune.analysis.multimodality import (
    compute_fitness_distance_correlation,
    compute_multimodality_metrics,
)
from exatune.analysis.smoothness import (
    compute_autocorrelation,
    compute_gradient_statistics,
    compute_ruggedness,
)


def compute_experiment_summary(
    df: pd.DataFrame,
    metric: str = "mean_score",
    param_columns: Optional[List[str]] = None,
    direction: str = "maximize",
) -> Dict[str, Any]:
    """Compute a comprehensive summary of the experiment landscape.

    Args:
        df: Results DataFrame.
        metric: Primary metric.
        param_columns: Hyperparameter columns.
        direction: Optimization direction ('maximize' or 'minimize').

    Returns:
        Dict with sections: overview, best_config, importance,
        smoothness, multimodality, interactions, timing.
    """
    filtered, metric_col, param_columns = validate_and_filter(df, metric, param_columns)

    # Overview
    values = filtered[metric_col].values
    total_in_df = len(df)
    successful = len(filtered)
    overview = {
        "n_configurations_total": total_in_df,
        "n_successful": successful,
        "success_rate": float(successful / total_in_df) if total_in_df > 0 else 0.0,
        "metric": metric_col,
        "direction": direction,
        "metric_mean": float(np.mean(values)),
        "metric_std": float(np.std(values)),
        "metric_min": float(np.min(values)),
        "metric_max": float(np.max(values)),
        "metric_median": float(np.median(values)),
    }

    # Best configuration
    if direction == "maximize":
        best_idx = int(np.argmax(values))
    else:
        best_idx = int(np.argmin(values))

    best_row = filtered.iloc[best_idx]
    best_config = {param: best_row[param] for param in param_columns}
    best_config["_score"] = float(best_row[metric_col])

    # Importance
    importance_df = compute_importance_summary(filtered, metric_col, param_columns)

    # Smoothness
    autocorr = compute_autocorrelation(filtered, metric_col, param_columns)
    gradient = compute_gradient_statistics(filtered, metric_col, param_columns)
    ruggedness = compute_ruggedness(filtered, metric_col, param_columns)

    smoothness = {
        "autocorrelation": autocorr,
        "gradient_statistics": gradient,
        "ruggedness": ruggedness,
    }

    # Multimodality
    multimodality = compute_multimodality_metrics(
        filtered, metric_col, direction, param_columns
    )
    fdc = compute_fitness_distance_correlation(
        filtered, metric_col, direction, param_columns
    )
    multimodality["fitness_distance_correlation"] = fdc

    # Interactions
    interactions = detect_significant_interactions(filtered, metric_col, param_columns)

    # Timing
    timing = {}
    if "fit_time_mean" in filtered.columns:
        fit_times = filtered["fit_time_mean"].dropna().values
        if len(fit_times) > 0:
            timing["mean_fit_time"] = float(np.mean(fit_times))
            timing["total_fit_time"] = float(np.sum(fit_times))
    if "score_time_mean" in filtered.columns:
        score_times = filtered["score_time_mean"].dropna().values
        if len(score_times) > 0:
            timing["mean_score_time"] = float(np.mean(score_times))

    return {
        "overview": overview,
        "best_config": best_config,
        "importance": importance_df.to_dict("records"),
        "smoothness": smoothness,
        "multimodality": multimodality,
        "interactions": [
            {"param_1": p1, "param_2": p2, "strength": s} for p1, p2, s in interactions
        ],
        "timing": timing,
    }


def format_summary_text(summary: Dict[str, Any]) -> str:
    """Format the experiment summary as a readable text report.

    Args:
        summary: Output from compute_experiment_summary().

    Returns:
        Formatted multi-line string report.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("EXATUNE LANDSCAPE ANALYSIS REPORT")
    lines.append("=" * 70)

    # Overview
    ov = summary["overview"]
    lines.append("")
    lines.append("OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"  Total configurations:  {ov['n_configurations_total']}")
    lines.append(f"  Successful:            {ov['n_successful']} ({ov['success_rate']:.1%})")
    lines.append(f"  Metric:                {ov['metric']} ({ov['direction']})")
    lines.append(f"  Mean:                  {ov['metric_mean']:.6f}")
    lines.append(f"  Std:                   {ov['metric_std']:.6f}")
    lines.append(f"  Min:                   {ov['metric_min']:.6f}")
    lines.append(f"  Max:                   {ov['metric_max']:.6f}")
    lines.append(f"  Median:                {ov['metric_median']:.6f}")

    # Best config
    bc = summary["best_config"]
    lines.append("")
    lines.append("BEST CONFIGURATION")
    lines.append("-" * 40)
    score = bc.pop("_score", None)
    if score is not None:
        lines.append(f"  Score: {score:.6f}")
    for param, val in bc.items():
        lines.append(f"  {param}: {val}")
    if score is not None:
        bc["_score"] = score

    # Importance
    lines.append("")
    lines.append("HYPERPARAMETER IMPORTANCE")
    lines.append("-" * 40)
    for entry in summary["importance"]:
        lines.append(
            f"  {entry['overall_rank']}. {entry['parameter']}: "
            f"variance={entry['variance_importance']:.4f}, "
            f"correlation={entry['correlation_importance']:.4f}, "
            f"range={entry['range_importance']:.4f}"
        )

    # Smoothness
    sm = summary["smoothness"]
    lines.append("")
    lines.append("LANDSCAPE SMOOTHNESS")
    lines.append("-" * 40)
    lines.append(f"  Moran's I:       {sm['autocorrelation']['morans_i']:.4f}")
    lines.append(f"  Ruggedness:      {sm['ruggedness']['ruggedness_index']:.4f}")
    lines.append(f"  Mean gradient:   {sm['gradient_statistics']['mean_gradient']:.6f}")
    lines.append(f"  Max gradient:    {sm['gradient_statistics']['max_gradient']:.6f}")
    lines.append(f"  Interpretation:  {sm['autocorrelation']['interpretation']}")

    # Multimodality
    mm = summary["multimodality"]
    lines.append("")
    lines.append("MULTIMODALITY")
    lines.append("-" * 40)
    lines.append(f"  Local optima:     {mm['n_local_optima']}")
    lines.append(f"  Optima density:   {mm['optima_density']:.4f}")
    lines.append(f"  Global optimum:   {mm['global_optimum_value']:.6f}")
    lines.append(f"  Funnel index:     {mm['funnel_index']:.4f}")
    fdc = mm.get("fitness_distance_correlation", {})
    if fdc:
        lines.append(f"  FDC:              {fdc.get('fdc', 0):.4f}")
    lines.append(f"  Interpretation:   {mm['interpretation']}")

    # Interactions
    lines.append("")
    lines.append("SIGNIFICANT INTERACTIONS")
    lines.append("-" * 40)
    if summary["interactions"]:
        for inter in summary["interactions"]:
            lines.append(
                f"  {inter['param_1']} x {inter['param_2']}: "
                f"strength={inter['strength']:.4f}"
            )
    else:
        lines.append("  No significant interactions detected.")

    # Timing
    if summary.get("timing"):
        lines.append("")
        lines.append("TIMING")
        lines.append("-" * 40)
        t = summary["timing"]
        if "mean_fit_time" in t:
            lines.append(f"  Mean fit time:   {t['mean_fit_time']:.3f}s")
        if "total_fit_time" in t:
            lines.append(f"  Total fit time:  {t['total_fit_time']:.1f}s")
        if "mean_score_time" in t:
            lines.append(f"  Mean score time: {t['mean_score_time']:.3f}s")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)
