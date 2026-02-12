"""Generate formatted analysis reports.

Produces markdown reports with optional companion visualization plots,
suitable for sharing experiment results and landscape analysis findings.
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from exatune.analysis._utils import infer_hyperparameter_columns, validate_and_filter
from exatune.analysis.summary import compute_experiment_summary, format_summary_text


def generate_report(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    metric: str = "mean_score",
    direction: str = "maximize",
    include_plots: bool = True,
    format: str = "markdown",
) -> Path:
    """Generate a comprehensive analysis report.

    Creates a markdown or text report with landscape analysis results.
    If include_plots is True, generates standard visualization plots
    alongside the report.

    Args:
        df: Results DataFrame.
        output_path: Directory to write report and plots.
        metric: Primary metric.
        direction: Optimization direction.
        include_plots: Whether to generate companion plots.
        format: 'markdown' or 'text'.

    Returns:
        Path to the generated report file.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    filtered, metric_col, param_columns = validate_and_filter(df, metric)

    summary = compute_experiment_summary(df, metric, param_columns, direction)

    if format == "markdown":
        content = _format_markdown_report(summary, param_columns)
        report_file = output_path / "landscape_report.md"
    else:
        content = format_summary_text(summary)
        report_file = output_path / "landscape_report.txt"

    report_file.write_text(content, encoding="utf-8")

    # Generate companion plots
    if include_plots:
        plots_dir = output_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        _generate_companion_plots(df, metric, param_columns, plots_dir)

    return report_file


def _format_markdown_report(summary: dict, param_columns: list) -> str:
    """Format summary as a markdown report."""
    lines = []
    lines.append("# ExaTune Landscape Analysis Report\n")

    # Overview
    ov = summary["overview"]
    lines.append("## Overview\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total configurations | {ov['n_configurations_total']} |")
    lines.append(f"| Successful | {ov['n_successful']} ({ov['success_rate']:.1%}) |")
    lines.append(f"| Optimization metric | {ov['metric']} ({ov['direction']}) |")
    lines.append(f"| Mean | {ov['metric_mean']:.6f} |")
    lines.append(f"| Std | {ov['metric_std']:.6f} |")
    lines.append(f"| Min | {ov['metric_min']:.6f} |")
    lines.append(f"| Max | {ov['metric_max']:.6f} |")
    lines.append(f"| Median | {ov['metric_median']:.6f} |")
    lines.append("")

    # Best config
    bc = summary["best_config"]
    lines.append("## Best Configuration\n")
    score = bc.get("_score")
    if score is not None:
        lines.append(f"**Score: {score:.6f}**\n")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    for param, val in bc.items():
        if param != "_score":
            lines.append(f"| {param} | {val} |")
    lines.append("")

    # Importance
    lines.append("## Hyperparameter Importance\n")
    lines.append("| Rank | Parameter | Variance | Correlation | Range |")
    lines.append("|------|-----------|----------|-------------|-------|")
    for entry in summary["importance"]:
        lines.append(
            f"| {entry['overall_rank']} | {entry['parameter']} "
            f"| {entry['variance_importance']:.4f} "
            f"| {entry['correlation_importance']:.4f} "
            f"| {entry['range_importance']:.4f} |"
        )
    lines.append("")

    # Smoothness
    sm = summary["smoothness"]
    lines.append("## Landscape Smoothness\n")
    lines.append(f"- **Moran's I**: {sm['autocorrelation']['morans_i']:.4f}")
    lines.append(f"- **Ruggedness index**: {sm['ruggedness']['ruggedness_index']:.4f}")
    lines.append(f"- **Mean gradient**: {sm['gradient_statistics']['mean_gradient']:.6f}")
    lines.append(f"- **Max gradient**: {sm['gradient_statistics']['max_gradient']:.6f}")
    lines.append(f"\n> {sm['autocorrelation']['interpretation']}")
    lines.append(f"\n> {sm['ruggedness']['interpretation']}")
    lines.append("")

    # Multimodality
    mm = summary["multimodality"]
    lines.append("## Multimodality\n")
    lines.append(f"- **Local optima**: {mm['n_local_optima']}")
    lines.append(f"- **Optima density**: {mm['optima_density']:.4f}")
    lines.append(f"- **Global optimum**: {mm['global_optimum_value']:.6f}")
    lines.append(f"- **Funnel index**: {mm['funnel_index']:.4f}")
    fdc = mm.get("fitness_distance_correlation", {})
    if fdc:
        lines.append(f"- **FDC**: {fdc.get('fdc', 0):.4f}")
        lines.append(f"\n> {fdc.get('interpretation', '')}")
    lines.append(f"\n> {mm['interpretation']}")
    lines.append("")

    # Interactions
    lines.append("## Significant Interactions\n")
    if summary["interactions"]:
        lines.append("| Parameter 1 | Parameter 2 | Strength |")
        lines.append("|-------------|-------------|----------|")
        for inter in summary["interactions"]:
            lines.append(
                f"| {inter['param_1']} | {inter['param_2']} "
                f"| {inter['strength']:.4f} |"
            )
    else:
        lines.append("No significant interactions detected.")
    lines.append("")

    # Timing
    if summary.get("timing"):
        t = summary["timing"]
        lines.append("## Timing\n")
        if "mean_fit_time" in t:
            lines.append(f"- **Mean fit time**: {t['mean_fit_time']:.3f}s")
        if "total_fit_time" in t:
            lines.append(f"- **Total fit time**: {t['total_fit_time']:.1f}s")
        if "mean_score_time" in t:
            lines.append(f"- **Mean score time**: {t['mean_score_time']:.3f}s")
        lines.append("")

    lines.append("---\n*Generated by ExaTune*\n")

    return "\n".join(lines)


def _generate_companion_plots(
    df: pd.DataFrame,
    metric: str,
    param_columns: list,
    plots_dir: Path,
) -> None:
    """Generate standard visualization plots alongside the report."""
    import matplotlib

    matplotlib.use("Agg")

    try:
        from exatune.visualization.dashboard import plot_dashboard
        from exatune.visualization.distributions import plot_score_histogram
        from exatune.visualization.importance import plot_importance
        from exatune.visualization.slices import plot_all_slices

        plot_score_histogram(df, metric=metric, output_path=plots_dir / "score_histogram.png")
        plot_importance(df, metric=metric, output_path=plots_dir / "importance.png")
        plot_all_slices(df, params=param_columns, metric=metric, output_path=plots_dir / "slices.png")

        if len(param_columns) >= 2:
            from exatune.visualization.landscape import plot_heatmap

            plot_heatmap(
                df,
                x_param=param_columns[0],
                y_param=param_columns[1],
                metric=metric,
                output_path=plots_dir / "heatmap.png",
            )

        plot_dashboard(df, metric=metric, params=param_columns, output_path=plots_dir / "dashboard.png")

    except Exception:
        pass  # Plots are optional; don't fail the report

    finally:
        import matplotlib.pyplot as plt

        plt.close("all")
